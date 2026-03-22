import copy
import math

import numpy as np
from typing import List, Tuple

from core.task import Task


def calculate_hyperperiod(tasks: List[Task]) -> int:
    """Calculate LCM of all task periods."""
    periods = [int(t.period) for t in tasks]
    lcm_val = periods[0]
    for p in periods[1:]:
        lcm_val = np.lcm(lcm_val, p)
    return int(lcm_val)


def get_normalization_utility_offline_stats(
        tasks: List[Task],
        drop_list: List[Task],
        mode: str = 'LO',
        grid_hyperperiod: int = None
) -> Tuple[float, float]:
    """
    计算离线理论归一化效用 (Weighted Executed Value, Weighted Total Value)。

    逻辑：
    1. 分母 (Total): 仅包含原任务 (Original Tasks) 在理想情况下的总价值。备份子块不贡献分母。
    2. 分子 (Executed):
       - LO Mode: 原任务运行 (m+x), 备份子块不运行。
       - HI Mode:
            - 被 Drop 的原任务停止 (0)。
            - 幸存的原任务运行 (m)。
            - 备份子块运行 (m, 即1)。
    """
    if not tasks: return 0.0, 0.0

    # 1. 确定时间窗口 (Hyperperiod)
    # 如果外部传入了全局 Hyperperiod，则使用全局的；否则计算局部的。
    if grid_hyperperiod is None:
        hyperperiod = calculate_hyperperiod(tasks)
    else:
        hyperperiod = grid_hyperperiod

    lo_total_value = 0.0
    lo_executed_value = 0.0

    # 为了快速查找，将 drop_list 转为 ID 集合 (假设 drop_list 里是原任务对象)
    drop_list_ids = {t.id for t in drop_list}

    for task in tasks:
        if task.criticality == "HI":
            continue  # HI 任务不贡献 Utility (假设 Utility 仅定义在 LO 任务上)

        # -----------------------------------------------------
        # 1. 计算分母 (Total Value)
        # -----------------------------------------------------
        num_jobs = hyperperiod / task.period

        # 【关键】仅原任务贡献分母，备份子块不贡献
        if not task.is_backup_subblock:
            lo_total_value += (num_jobs * task.utility)

        # -----------------------------------------------------
        # 2. 计算分子 (Executed Value)
        # -----------------------------------------------------

        effective_m = 0

        if mode == 'LO':
            # === LO 模式策略 ===
            if task.is_backup_subblock:
                # 备份子块: LO 模式下休眠
                effective_m = 0
            else:
                # 原任务: LO 模式下正常运行，享受 x 优化
                # (即使它在 HI 模式会被 Drop，但在 LO 模式它是正常工作的)
                effective_m = task.mk.m + task.mk.x

        else:  # mode == 'HI'
            # === HI 模式策略 ===
            #getattr(task, 'is_backup_subblock', False)
            if task.is_backup_subblock:
                # 备份子块: HI 模式下激活，运行 (1, k)
                effective_m = task.mk.m

            elif task.id in drop_list_ids:
                # 原任务 (且在 Drop List 中): HI 模式下停止
                effective_m = 0

            else:
                # 原任务 (幸存者): HI 模式下退回基础 m
                effective_m = task.mk.m

        # 计算在 effective_m 约束下的实际执行次数
        # (m,k) 模式计算公式
        if effective_m > 0:
            k = task.mk.k
            full_cycles = math.floor(num_jobs / k)
            remainder = num_jobs % k

            # 在一个 Hyperperiod 内执行的 mandatory 作业总数
            count = full_cycles * effective_m + min(remainder, effective_m)

            lo_executed_value += (count * task.utility)

    return lo_executed_value, lo_total_value
