from typing import List, Tuple, Callable, Any

from core.processor import Processor
from core.task import Task
from scheduling.priority_assignment import assign_static_priorities
from scheduling.sched_test import schedulability_test
from scheduling.task_partitioning import partition_step1_initial_assignment, partition_step3_reassign_subblocks


def cost_utility_utilization(task: Task) -> float:
    """
    Large Utility/Utilization First .
    """
    if task.wcet_lo == 0: return float('inf')
    return task.utility*task.period / task.wcet_lo

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

    流程逻辑：
    1. 【初始分配】一次性分配 HI 和 LO 任务 (WFD)。
    2. 【筛选丢弃】若核心不可调度，按 Cost 最低优先筛选 LO 任务移出。
    3. 【拆分重分配】将移出的任务拆分为 (1,k) 子块，按 Cost 最高优先尝试填回任意核心。
    4. 【效用优化】对幸存的完整 LO 任务优化 x 参数。
    5. 【状态固化】生成每个核心的 drop_list (用于在线模拟器在 HI 模式下丢弃 LO 任务)。

    Args:
        original_tasks: 原始任务列表。
        num_processors: 处理器核心数量。
        cost_func: 价值评估函数 (如 utility/wcet)。
                   - Drop 阶段: 越小越先被丢。
                   - Split/Optimize 阶段: 越大越优先保留/优化。
        is_dynamic_strategy: 是否在优化 x 阶段使用动态重排序策略。

    Returns:
        (is_success, processors)
        - is_success: 如果所有核心的 HI 任务都可调度，返回 True；否则 False。
        - processors: 包含最终任务分配和 drop_list 的处理器列表。
    """

    # 1. 初始分配
    processors = partition_step1_initial_assignment(original_tasks, num_processors)
    if processors is None: return False, []

    tasks_to_be_backed_up = []

    # 2. 调度保障:测试每个处理器
    for p in processors:

        while not schedulability_test(p.tasks, drop_task=p.drop_list):

            candidates = [t for t in p.tasks
                          if t.criticality == "LO"
                          and t not in p.drop_list
                          and not t.is_backup_subblock]

            if not candidates: return False, processors

            candidates.sort(key=cost_func)

            victim = candidates[0]

            # 原地标记 Drop (LO 继续跑，HI 停止)
            p.mark_as_dropped(victim)

            # 加入待备份队列
            tasks_to_be_backed_up.append(victim)

            assign_static_priorities(p.tasks)

    # 3. 备份子块分配 (All-or-Nothing)
    # 只有全部子块都能分配成功的任务，其子块才会出现在 processors 中
    _,_ = partition_step3_reassign_subblocks(
        tasks_to_split=tasks_to_be_backed_up,
        processors=processors,
        cost_func=cost_func
    )

    # 4. 效用优化 (Optimize X)
    for p in processors:
        # 仅优化：LO 任务，非备份块，且未被 Drop
        candidates = [t for t in p.tasks
                      if t.criticality == "LO"
                      and not t.is_backup_subblock]

        if not candidates: continue

        upgradable = list(candidates)
        upgradable.sort(key=cost_func, reverse=True)

        while upgradable:
            if is_dynamic_strategy:
                upgradable.sort(key=cost_func) #这里的动态策略主要指fair-min-max
            target = upgradable[0]
            # 尝试增加 x
            if target.mk.m + target.mk.x < target.mk.k:
                target.mk.increase_x(1)
                if not schedulability_test(p.tasks, drop_task=p.drop_list):
                    target.mk.increase_x(-1)
                    upgradable.remove(target)  # 该任务已达当前系统的极限
            else:
                upgradable.remove(target)
    return True, processors

