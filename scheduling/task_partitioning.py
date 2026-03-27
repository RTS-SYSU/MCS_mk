import copy
from typing import List, Callable, Optional, Tuple

from core.processor import Processor
from core.task import Task, MKPattern
from scheduling.priority_assignment import assign_static_priorities
from scheduling.sched_test import schedulability_test


def partition_tasks_wfd(original_tasks: List[Task],
                        num_processors: int,
                        sched_test_func: Callable[[List[Task]], bool]) -> Optional[List[Processor]]:
    """
    Partitions tasks across multiple processors using the Worst-Fit Decreasing (WFD) algorithm.

    This function is agnostic to the specific schedulability test used. When each task is allocated, the shcedulability test will be performed.

    :param original_tasks: List of tasks to be partitioned.
    :param num_processors: Total number of available processors.
    :param sched_test_func: A callback function that accepts a list of tasks and returns True
                            if they are schedulable, False otherwise.
    :return: A list of Processor objects with assigned tasks, or None if partitioning fails.
    """

    # Initialize processors
    processors = [Processor(i) for i in range(num_processors)]

    hi_tasks = [t for t in original_tasks if t.criticality == "HI"]
    lo_tasks = [t for t in original_tasks if t.criticality == "LO"]

    # --- 1. HI 任务分配 ---
    # 策略：按 HI 利用率降序
    hi_tasks.sort(key=lambda t: t.wcet_hi / t.period, reverse=True)

    for task in hi_tasks:

        processors.sort(key=lambda p: p.utilization_hi)
        assigned = False

        for p in processors:
            # Create a candidate task list for this processor
            candidate_tasks = p.tasks + [task]

            assign_static_priorities(candidate_tasks)

            if sched_test_func(candidate_tasks):
                p.add_task(task)
                # Persist the priority update for the processor's task list
                assign_static_priorities(p.tasks)
                assigned = True
                break

        if not assigned:
            #print(f"Error: Task {task.id} could not be assigned to any processor under WFD.")
            return None  # Partitioning failed


        # --- 2. LO 任务分配 ---
        # 策略：按有效利用率 ((m/k)*C_LO/T) 降序
    lo_tasks.sort(key=lambda t: (t.mk.m / t.mk.k) * (t.wcet_lo / t.period), reverse=True)

    for task in lo_tasks:

        processors.sort(key=lambda p:p.utilization_lo)
        assigned = False

        # 预先计算该 LO 任务带来的利用率增量
        # LO 任务在 HI 模式下的贡献按照 (m/k)*C_LO 计算
        for p in processors:

            candidate_tasks = p.tasks + [task]
            assign_static_priorities(candidate_tasks)

            if sched_test_func(candidate_tasks):
                p.add_task(task)
                # Persist the priority update for the processor's task list
                assign_static_priorities(p.tasks)
                assigned = True
                break

        if not assigned:
            # print(f"[Partition Fail] LO Task {task.id} fits nowhere. (U>1)")
            return None

    return processors

# --- Step 1: 初始分配 ---
def partition_initial_assignment(original_tasks: List[Task], num_processors: int) -> Optional[List[Processor]]:
    """
    【Step 1】一次性分配 HI 和 LO 任务到处理器。
    如果在分配过程中，任务无法放入任何核心（导致 U_LO > 1.0 或 U_HI > 1.0），则返回 None。
    """
    processors = [Processor(i) for i in range(num_processors)]

    hi_tasks = [t for t in original_tasks if t.criticality == "HI"]
    lo_tasks = [t for t in original_tasks if t.criticality == "LO"]

    # --- 1. HI 任务分配 ---
    # 策略：按 HI 利用率降序
    hi_tasks.sort(key=lambda t: t.wcet_hi / t.period, reverse=True)

    for task in hi_tasks:
        # WFD: 尝试放入 HI 负载最小的核心
        processors.sort(key=lambda p: p.utilization_hi)

        assigned = False

        # 预先计算该 HI 任务带来的利用率增量
        u_inc_lo = task.wcet_lo / task.period
        u_inc_hi = task.wcet_hi / task.period

        for p in processors:
            # 检查加入后是否超限
            if (p.utilization_lo + u_inc_lo <= 1.0) and (p.utilization_hi + u_inc_hi <= 1.0):
                p.add_task(task)
                assigned = True
                break

        if not assigned:
            #print(f"[Partition Fail] HI Task {task.id} fits nowhere. (U>1)")
            return None

    # --- 2. LO 任务分配 ---
    # 策略：按有效利用率 ((m/k)*C_LO/T) 降序
    lo_tasks.sort(key=lambda t: (t.mk.m / t.mk.k) * (t.wcet_lo / t.period), reverse=True)

    for task in lo_tasks:

        processors.sort(key=lambda p: p.utilization_lo)

        assigned = False

        # 预先计算该 LO 任务带来的利用率增量
        # LO 任务在 HI 模式下的贡献按照 (m/k)*C_LO 计算
        ratio = task.mk.m / task.mk.k
        u_inc_lo = (task.wcet_lo / task.period) * ratio
        u_inc_hi = (task.wcet_lo / task.period) * ratio

        for p in processors:
            # 检查加入后是否超限
            if (p.utilization_lo + u_inc_lo <= 1.0) and (p.utilization_hi + u_inc_hi <= 1.0):
                p.add_task(task)
                assigned = True
                break

        if not assigned:
            #print(f"[Partition Fail] LO Task {task.id} fits nowhere. (U>1)")
            return None

    # 预分配优先级
    for p in processors:
        assign_static_priorities(p.tasks)

    return processors

# splitting and migration mapping
def partition_reassign_subtasks(
        tasks_to_split: List[Task],
        processors: List[Processor],
        cost_func: Callable[[Task], float]
) -> Tuple[List[Task], List[Task]]:
    """
    将待丢弃的任务进行拆分，并尝试分配备份子块。
    【约束】原子性分配：对于一个任务，必须所有 m 个子块都成功分配，否则全部撤销。
    """

    # 1. 准备工作：待处理任务队列
    # 我们以"完整任务"为单位进行处理，而不是以"子块"为单位
    pending_tasks = list(tasks_to_split)

    # 2. 初始排序：优先挽救高价值任务
    pending_tasks.sort(key=cost_func, reverse=True)


    success_subblocks_total = []
    failed_tasks_total = []

    ID_MULTIPLIER = 1000

    # 3. 逐个任务尝试
    while pending_tasks:

        # 动态策略：每轮重新评估任务优先级,这次指fair-min-max，就是按照mk来，偏随机性，不需要强调

        current_task = pending_tasks.pop(0)
        m_orig = current_task.mk.m
        k_orig = current_task.mk.k

        # --- 事务开始 ---
        current_task_subblocks = []
        transaction_log = {}  # 记录 [(processor:subblock), ...] 以便回滚

        # 生成该任务的所有子块
        for j in range(m_orig):
            sb = copy.deepcopy(current_task)
            sb.id = int(current_task.id * ID_MULTIPLIER + j)
            sb.parent_id = current_task.id
            sb.is_backup_subblock = True  # 标记为仅 HI 模式运行

            sb.wcet_lo = current_task.wcet_lo
            sb.wcet_hi = current_task.wcet_hi

            sb.mk = MKPattern(m=1, k=k_orig, offset=j)
            current_task_subblocks.append(sb)

        # 尝试分配所有子块
        task_success = True

        for sb in current_task_subblocks:
            assigned = False

            # sub-tasks will be activated in HI mode
            processors.sort(key=lambda p: p.utilization_hi)

            for p in processors:
                # 试探性加入
                candidate_tasks = p.tasks + [sb]
                assign_static_priorities(candidate_tasks)

                if schedulability_test(candidate_tasks, drop_task=p.drop_list):
                    # assigned success
                    assigned = True
                    if p in transaction_log:
                        existing_sb = transaction_log[p]
                        merged_mk = existing_sb.mk.merge_pattern(sb.mk)
                        existing_sb.mk = merged_mk
                    else:
                        p.add_task(sb)
                        assign_static_priorities(p.tasks)
                        # 记录到事务日志
                        transaction_log[p]=sb
                    break

            if not assigned:
                task_success = False
                break  # 只要有一个子块失败，整个任务分配失败

        # --- 事务提交或回滚 ---
        if task_success:
            # Commit
            for sb in transaction_log.values():
                success_subblocks_total.append(sb)
            # print(f"Task {current_task.id} backup success ({m_orig} blocks).")
        else:
            # Rollback
            # 移除所有已分配的子块
            for p, sb in transaction_log.items():
                p.remove_task(sb)  # remove_task 会自动扣减利用率
                assign_static_priorities(p.tasks)

            failed_tasks_total.append(current_task)
            # print(f"Task {current_task.id} backup failed (Rolled back).")

    return success_subblocks_total, failed_tasks_total
