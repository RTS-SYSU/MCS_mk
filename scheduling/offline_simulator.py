import copy
from typing import List, Tuple, Callable, Any

from core.processor import Processor
from core.task import Task
from scheduling.priority_assignment import assign_static_priorities
from scheduling.sched_test import schedulability_test, calculate_wcrt_lo, test_aMC
from scheduling.task_partitioning import partition_reassign_subtasks, partition_filter, partition_only


def cost_utility_utilization(task: Task) -> float:
    """
    Large Utility/Utilization First .
    """
    if task.wcet_lo == 0: return float('inf')
    return task.utility * task.period / task.wcet_lo


def cost_utility_density(task: Task) -> float:
    """
        Large Utility/WCET First .
        """
    if task.wcet_lo == 0: return float('inf')
    return task.utility / task.wcet_lo


def cost_high_util(task: Task) -> float:
    if task.period == 0: return float('inf')
    return task.wcet_lo / task.period


def cost_low_util(task: Task) -> float:
    if task.period == 0: return float('inf')
    return -(task.wcet_lo / task.period)


def cost_value_density(task: Task) -> float:
    return task.wcet_lo


def cost_dynamic_fairness(task: Task) -> float:
    """
    Dynamic Fairness / Min-Max Utility.
    Returns current utility ratio (m+x)/k.
    Used for iterative sorting: Tasks with the lowest current utility get priority.
    """
    if not task.mk or task.mk.k == 0: return float('inf')
    # Dynamic Calculation based on current x
    return (task.mk.m + task.mk.x) / task.mk.k


def uaswc_offline_multicore(
        original_tasks: List[Task],
        num_processors: int,
        cost_func: Callable[[Task], float],
        is_dynamic_strategy: bool = False
) -> Tuple[bool, list[Processor]]:
    """
    UASWC 多核离线优化主流程。

    Args:
        original_tasks: 原始任务列表。
        num_processors: 处理器核心数量。
        cost_func: 价值评估函数。
                   - Drop 阶段: 不可调度被drop。
                   - Split/Optimize 阶段: 越大越优先保留/优化。
        is_dynamic_strategy: 是否在优化 x 阶段使用动态重排序策略。

    Returns:
        (is_success, processors)
        - is_success: 如果所有核心的 HI 任务都可调度，返回 True；否则 False。
        - processors: 包含最终任务分配和 drop_list 的处理器列表。
    """

    # 1. init partitioning
    # processors = partition_filter(original_tasks, num_processors)
    # if processors is None: return False, []
    #
    # split_tasks = []
    # for p in processors:
    #     split_tasks.extend(p.drop_list)
    # assert len(split_tasks) == len({t.id for t in split_tasks}), "Duplicate tasks in split_tasks!"

    # ==========================================================================
    processors = partition_only(original_tasks, num_processors)
    if processors is None: return False, []
    split_tasks = []

    for p in processors:
        p.tasks.sort(key=lambda t: t.priority)
        for task in p.tasks:
            R_lo, ok_lo = calculate_wcrt_lo(task, p.tasks)
            if not ok_lo:
                return False, processors

            ok_mc = test_aMC(task, p.tasks, R_lo, p.drop_list)
            if not ok_mc:
                if task.criticality == "LO":
                    p.mark_as_dropped(task)
                    split_tasks.append(task)
                else:
                    # HI tasks must meet deadline
                    return False, processors

    # 3. 备份子块分配 (All-or-Nothing)
    # 只有全部子块都能分配成功的任务，其子块才会出现在 processors 中
    success_subblocks, failed_tasks = partition_reassign_subtasks(
        tasks_to_split=split_tasks,
        processors=processors,
        cost_func=cost_func
    )
    # if len(success_subblocks) > 0:
    #     print("sub-tasks:",len(success_subblocks)," ", success_subblocks)
    #     print("failed_tasks:",len(failed_tasks)," ", failed_tasks)
    #     for p in processors:
    #         print("p:",p.tasks)

    # 4. 效用优化 (Optimize X)
    for p in processors:
        # LO task is optimized, but no sub-task and drop task)
        candidates = [t for t in p.tasks
                      if t.criticality == "LO"
                      and t not in p.drop_list
                      and not t.is_backup_subblock]

        if not candidates: continue

        upgradable = list(candidates)
        #candidates.sort(key=cost_func, reverse=True)
        upgradable.sort(key=cost_func, reverse=True)
        # degradable = copy.deepcopy(candidates)


        while upgradable:
            if is_dynamic_strategy:
                upgradable.sort(key=cost_func)  # 这里的动态策略主要指fair-min-max
            target = upgradable[0]
            # 尝试增加 x
            if target.mk.m + target.mk.x < target.mk.k:
                target.mk.increase_x(1)
                if not schedulability_test(p.tasks, drop_task=p.drop_list):
                    target.mk.increase_x(-1)
                    upgradable.remove(target)  # 该任务已达当前系统的极限
            else:
                upgradable.remove(target)



        degradable = list(candidates)
        degradable.sort(key=cost_func)

        for de_task in degradable:
            dex = de_task.mk.x
            de_task.mk.dx = dex
            de_task.mk.update_pattern(is_degraded=True)

        while degradable:

            if schedulability_test(p.tasks, drop_task=p.drop_list):
                break

            if is_dynamic_strategy:
                degradable.sort(key=lambda t: (t.mk.m + t.mk.dx) / t.mk.k, reverse=True)
            target = degradable[0]
            dx = target.mk.dx
            if dx > 0:
                target.mk.increase_x(-1, is_degraded=True)
            else:
                degradable.remove(target)

                # Phase 2: Find HI-mode safe maximum x
        # candidates.sort(key=cost_func, reverse=True)

        # for de_task in degradable:
        #     dx = de_task.mk.x  # 离线升级后的 x
        #     # 从高到低尝试，找到 HI-mode 也能通过的最大 x
        #     while dx >= 0:
        #         de_task.mk.dx = dx
        #         de_task.mk.update_pattern(is_degraded=True)
        #         if schedulability_test(p.tasks, drop_task=p.drop_list):
        #             break
        #         dx -= 1

    return True, processors
