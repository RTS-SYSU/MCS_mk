import copy
from typing import List, Callable, Optional, Tuple

from core.processor import Processor
from core.task import Task, MKPattern
from scheduling.priority_assignment import assign_static_priorities
from scheduling.sched_test import schedulability_test, test_aLO, calculate_wcrt_lo, test_aMC


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
        u_inc_lo = task.wcet_lo / task.period
        u_inc_hi = task.wcet_hi / task.period
        for p in processors:
            # Create a candidate task list for this processor
            candidate_tasks = p.tasks + [task]

            assign_static_priorities(candidate_tasks)

            if (p.utilization_lo + u_inc_lo <= 1.0) and (p.utilization_hi + u_inc_hi <= 1.0) and sched_test_func(
                    candidate_tasks):
                p.add_task(task)
                # Persist the priority update for the processor's task list
                assign_static_priorities(p.tasks)
                assigned = True
                break

        if not assigned:
            # print(f"Error: Task {task.id} could not be assigned to any processor under WFD.")
            return None  # Partitioning failed

        # --- 2. LO 任务分配 ---
        # 策略：按有效利用率 ((m/k)*C_LO/T) 降序
    lo_tasks.sort(key=lambda t: (t.mk.m / t.mk.k) * (t.wcet_lo / t.period), reverse=True)

    for task in lo_tasks:

        processors.sort(key=lambda p: p.utilization_lo)
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


def partition_filter(original_tasks: List[Task], num_processors: int) -> Optional[List[Processor]]:
    """
    partitioning and filter unschedulable task in drop task set
    HI task first, and then LO task.
    if HI task unschedulable in any mode and LO task unschedulable in LO mode, partitioning fails return False.
    if LO task unschedulable in HI mode, put the task into drop task set and then splitting and migration.
    """
    processors = [Processor(i) for i in range(num_processors)]

    hi_tasks = [t for t in original_tasks if t.criticality == "HI"]
    lo_tasks = [t for t in original_tasks if t.criticality == "LO"]

    # --- 1. HI 任务分配 ---
    # 策略：按 HI 利用率降序
    hi_tasks.sort(key=lambda t: t.wcet_hi / t.period, reverse=True)

    for curr_task in hi_tasks:
        # WFD: 尝试放入 HI 负载最小的核心
        processors.sort(key=lambda p: p.utilization_hi)

        assigned = False

        # 预先计算该 HI 任务带来的利用率增量
        u_inc_lo = curr_task.wcet_lo / curr_task.period
        u_inc_hi = curr_task.wcet_hi / curr_task.period

        for p in processors:
            # 检查加入后是否超限
            candidate_tasks = p.tasks + [curr_task]

            assign_static_priorities(candidate_tasks)
            # every HI task must be scheduled in both mode.
            # schedulability_test(candidate_tasks): if you want, you can split this to reduce time complexity, like LO task (can't avoid)
            if (p.utilization_lo + u_inc_lo <= 1.0) and (p.utilization_hi + u_inc_hi <= 1.0) and schedulability_test(
                    candidate_tasks):
                p.add_task(curr_task)
                # assign_static_priorities(p.tasks)  #tasks priority also can be assigned after partitioning
                assigned = True
                break

        if not assigned:
            # print(f"[Partition Fail] HI Task {task.id} fits nowhere. (U>1)")
            return None

    # --- 2. LO 任务分配 ---
    # 策略：按有效利用率 ((m/k)*C_LO/T) 降序
    lo_tasks.sort(key=lambda t: (t.mk.m / t.mk.k) * (t.wcet_lo / t.period), reverse=True)

    for curr_task in lo_tasks:

        processors.sort(key=lambda p: p.utilization_lo)

        assigned = False

        # 预先计算该 LO 任务带来的利用率增量
        # LO 任务在 HI 模式下的贡献按照 (m/k)*C_LO 计算
        ratio = curr_task.mk.m / curr_task.mk.k
        u_inc_lo = (curr_task.wcet_lo / curr_task.period) * ratio
        #u_inc_hi = (curr_task.wcet_lo / curr_task.period) * ratio

        for p in processors:

            if p.utilization_lo + u_inc_lo > 1.0:
                continue

            candidate_tasks = p.tasks + [curr_task]
            assign_static_priorities(candidate_tasks)
            lp_tasks = [t for t in candidate_tasks if t.priority > curr_task.priority]  # lower priority task set than current task

            lp_schedulable = True

            for test_task in lp_tasks:
                R_lo, ok_lo = calculate_wcrt_lo(test_task, candidate_tasks)
                if ok_lo:
                    # due to R_mc dominates R_hi, only verifying R_mc suffices to ensure schedulability in HI mode.
                    ok_mc = test_aMC(test_task, candidate_tasks, R_lo, drop_task=p.drop_list)

                    if not ok_mc:
                        lp_schedulable = False
                        break
                else:
                    lp_schedulable = False
                    break

            # lower priority tasks unschedulable, switch to next processor core
            if not lp_schedulable:
                continue

            R_lo, ok_lo = calculate_wcrt_lo(curr_task, candidate_tasks)
            if not ok_lo:
                continue

            p.add_task(curr_task)
            ok_mc = test_aMC(curr_task, candidate_tasks, R_lo, drop_task=p.drop_list)

            if not ok_mc:
                # if LO task unschedulable in HI mode, put the task into drop task set and then splitting and migration.
                p.mark_as_dropped(curr_task)

            assigned = True
            break

        if not assigned:
            # print(f"[Partition Fail] LO Task {task.id} fits nowhere. (U>1)")
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
        transaction_log = []  # 记录 [(processor,subblock), ...] 以便回滚

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
                candidate_tasks = p.tasks + [sb]
                assign_static_priorities(candidate_tasks)

                if schedulability_test(candidate_tasks, drop_task=p.drop_list):
                    assigned = True
                    p.add_task(sb)  # 内部会自动更新利用率
                    assign_static_priorities(p.tasks)
                    transaction_log.append((p,sb))
                    break

            if not assigned:
                task_success = False
                break  # 只要有一个子块失败，整个任务分配失败
        # --- 事务提交或回滚 ---
        if task_success:
            # Commit
            # Step 1: 按处理器分组已分配的子块
            # { processor: [subblock1, subblock2, ...] }
            core_subblocks = {}
            for p, sb in transaction_log:
                if p not in core_subblocks:
                    core_subblocks[p] = []
                core_subblocks[p].append(sb)

            # Step 2: 对每个处理器上的子块，按 parent_id 分组并合并
            for p, subblocks in core_subblocks.items():
                # 按 parent_id 分组
                # { parent_id: [subblock1, subblock2, ...] }
                task_groups = {}
                for sb in subblocks:
                    if sb.parent_id not in task_groups:
                        task_groups[sb.parent_id] = []
                    task_groups[sb.parent_id].append(sb)

                # 对每个任务组的子块进行合并
                for parent_id, sb_list in task_groups.items():
                    if len(sb_list) == 1:
                        # 只有一个子块，无需合并
                        success_subblocks_total.append(sb_list[0])
                    else:
                        # 多个子块，合并为一个
                        # 以第一个子块为基础
                        merged_sb = copy.deepcopy(sb_list[0])
                        merged_mk = sb_list[0].mk

                        # 依次合并其他子块的 pattern
                        for sb in sb_list[1:]:
                            merged_mk = merged_mk.merge_pattern(sb.mk)
                            # 从处理器任务列表中移除原子块
                            p.remove_task(sb)
                        p.remove_task(sb_list[0])

                        # 更新合并后的子块
                        merged_sb.mk = merged_mk
                        # 重新计算利用率 (m 个 mandatory jobs)
                        merged_sb.wcet_hi = merged_sb.wcet_lo  # 备份子块 WCET 不变
                        u_inc = (merged_sb.wcet_hi / merged_sb.period) * (merged_mk.m / merged_mk.k)

                        # 将合并后的子块加入处理器
                        p.add_task(merged_sb)
                        assign_static_priorities(p.tasks)

                        # 加入成功列表
                        success_subblocks_total.append(merged_sb)
            #success_subblocks_total.extend(current_task_subblocks)
            # print(f"Task {current_task.id} backup success ({m_orig} blocks).")
        else:
            # Rollback
            # 移除所有已分配的子块
            for p, sb in transaction_log:
                p.remove_task(sb)  # remove_task 会自动扣减利用率
                assign_static_priorities(p.tasks)

            failed_tasks_total.append(current_task)
            # print(f"Task {current_task.id} backup failed (Rolled back).")

    return success_subblocks_total, failed_tasks_total

def partition_only(original_tasks: List[Task], num_processors: int) -> Optional[List[Processor]]:
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

def partition_only_other(original_tasks: List[Task], num_processors: int) -> Optional[List[Processor]]:
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
    lo_tasks.sort(key=lambda t:  t.wcet_lo / t.period, reverse=True)

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