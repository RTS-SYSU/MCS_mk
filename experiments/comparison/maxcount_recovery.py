r"""
MaxCount Recovery method.

Same LO-mode augmentation and mode-switch degradation as the proposed method.

Recovery phase: considers suspended tasks in nondecreasing order of their
LO-mode utilization u_i^LO = C_i^LO / T_i (low-utilization-first).  This
favours tasks with smaller execution demand and approximates a maximum-
recovery-count strategy without explicitly accounting for importance loss.
"""

from typing import Dict, List

from core.processor import Processor
from core.task import Task
from scheduling.augmentation import lo_mode_augment, mode_switch_degrade
from scheduling.recovery import (
    _build_sacrifice_sequence,
    _feasible_recovery,
    _binary_search_feasible,
    _commit_recovery,
)
from experiments.performance import (
    global_max_importance,
    core_importance,
)


def run(processors: List[Processor],
        all_lo_tasks: List[Task],
        beta: float = 0.5) -> Dict[str, float]:
    """
    Apply MaxCount Recovery to a partitioned system.

    Modifies processors in place.

    Returns {mode: Perf^mode}.
    """
    # LO-mode augmentation (same as ours)
    for p in processors:
        lo_mode_augment(p.tasks, drop_list=p.drop_list, beta=beta)

    # Snapshot MC drop_list
    drops_mc = [list(p.drop_list) for p in processors]

    # Mode-switch degradation (same as ours)
    for p in processors:
        mode_switch_degrade(p.tasks, drop_list=p.drop_list, beta=beta)

    # --- MaxCount Recovery (per core, low-utilization-first ordering) ---
    for p in processors:
        _maxcount_recover_core(p.tasks, p.drop_list, beta)

    # (No post-augment: MaxCount only recovers, does not augment x_h)

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


# ---------------------------------------------------------------------------
# MaxCount per-core recovery
# ---------------------------------------------------------------------------

def _maxcount_recover_core(tasks: List[Task],
                           drop_list: List[Task],
                           beta: float) -> None:
    """
    Recover suspended LO tasks in low-utilization-first order.
    """
    # Init x_h for retained, 0 for suspended
    for t in tasks:
        if t.criticality == "LO" and t not in drop_list:
            t.mk.x_h = t.mk.x_s
        elif t.criticality == "LO":
            t.mk.x_h = 0

    suspended = [t for t in drop_list if t.criticality == "LO"]
    if not suspended:
        return

    # Sort by u_i^LO = C_LO / T ascending (low utilization first)
    suspended.sort(key=lambda t: t.wcet_lo / t.period)

    for candidate in suspended:
        if candidate not in drop_list:
            continue

        sequence = _build_sacrifice_sequence(candidate, tasks, drop_list, beta)

        if not sequence:
            if _feasible_recovery(candidate, tasks, drop_list, [], 0):
                _commit_recovery(candidate, tasks, drop_list, [], 0)
            continue

        q_total = len(sequence)

        q_min = _binary_search_feasible(candidate, tasks, drop_list, sequence, q_total)
        if q_min is None:
            continue

        _commit_recovery(candidate, tasks, drop_list, sequence, q_min)
