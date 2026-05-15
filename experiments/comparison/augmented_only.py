r"""
Augmented-Only method.

Enables LO-mode augmentation and mode-switch degradation, but disables
stable-HI recovery: Gamma_LO^rec = empty.

In stable HI mode, available slack is used only to preserve or improve
the augmented levels of retained LO tasks (post-augmentation).
"""

from typing import Dict, List

from core.processor import Processor
from core.task import Task
from scheduling.augmentation import lo_mode_augment, mode_switch_degrade
from scheduling.recovery import _hi_mode_augment
from experiments.performance import (
    global_max_importance,
    core_importance,
)


def run(processors: List[Processor],
        all_lo_tasks: List[Task],
        beta: float = 0.5) -> Dict[str, float]:
    """
    Apply Augmented-Only to a partitioned system.

    Modifies processors in place.

    Returns {mode: Perf^mode}.
    """
    # LO-mode augmentation (same as ours)
    for p in processors:
        lo_mode_augment(p.tasks, drop_list=p.drop_list, beta=beta)

    # Snapshot MC drop_list before degradation (for Perf^S)
    drops_mc = [list(p.drop_list) for p in processors]

    # Mode-switch degradation (same as ours)
    for p in processors:
        mode_switch_degrade(p.tasks, drop_list=p.drop_list, beta=beta)

    # Stable HI mode: post-augment only, NO recovery
    for p in processors:
        _hi_mode_augment(p.tasks, p.drop_list, beta)

    # Compute performance
    i_max = global_max_importance(all_lo_tasks, beta)
    if i_max == 0:
        return {'L': 0.0, 'S': 0.0, 'H': 0.0}

    total_l = sum(core_importance(p.tasks, [], beta, 'L') for p in processors)
    total_s = sum(core_importance(p.tasks, drops_mc[p.id], beta, 'S')
                  for p in processors)
    total_h = sum(core_importance(p.tasks, p.drop_list, beta, 'H')
                  for p in processors)

    return {
        'L': total_l / i_max,
        'S': total_s / i_max,
        'H': total_h / i_max,
    }
