r"""
Performance metrics for the augmented weakly-hard importance model.

For each mode chi in {L, S, H}:
    I^chi = sum_{tau_i in Gamma_LO} a_i^chi * I_i(x_i^chi)

    where I_i(x) = mu_i * (1 + beta * x / (k_i - m_i))

Normalized performance:
    Perf^chi = I^chi / I^max

    where I^max = sum_{tau_i in Gamma_LO} I_i(k_i - m_i)
                = (1 + beta) * sum mu_i
"""

from typing import Dict, List

from core.processor import Processor
from core.task import Task, Mode


# ---------------------------------------------------------------------------
# I^max — maximum attainable importance (all LO tasks active, fully augmented)
# ---------------------------------------------------------------------------

def global_max_importance(lo_tasks: List[Task], beta: float) -> float:
    """
    I^max = sum_{tau_i in Gamma_LO} I_i(k_i - m_i)
          = (1 + beta) * sum mu_i
    """
    return sum(t.baseline_importance for t in lo_tasks) * (1.0 + beta)


# ---------------------------------------------------------------------------
# I^chi for a single core
# ---------------------------------------------------------------------------

def core_importance(
    tasks: List[Task],
    drop_list: List[Task],
    beta: float,
    mode: Mode = 'L',
) -> float:
    """
    I^chi for one core.

    LO mode ('L'): all LO tasks active regardless of drop_list.
    Mode switch ('S') and HI mode ('H'): tasks in drop_list are inactive (a_i = 0).
    """
    total = 0.0
    drop_ids = {t.id for t in drop_list}
    for t in tasks:
        if t.criticality == "HI":
            continue
        if mode != 'L' and t.id in drop_ids:
            continue
        total += t.calculate_importance(beta, mode)
    return total


# ---------------------------------------------------------------------------
# All-core normalized performance
# ---------------------------------------------------------------------------

def normalized_performance(
    processors: List[Processor],
    all_lo_tasks: List[Task],
    beta: float,
    mode: Mode = 'L',
) -> float:
    """
    Perf^chi = (sum over cores I^chi) / I^max
    """
    i_max = global_max_importance(all_lo_tasks, beta)
    if i_max == 0:
        return 0.0
    total = sum(core_importance(p.tasks, p.drop_list, beta, mode) for p in processors)
    return total / i_max


def all_performances(
    processors: List[Processor],
    all_lo_tasks: List[Task],
    beta: float,
) -> Dict[str, float]:
    """
    Return {mode: Perf^mode} for L, S, H.

    NOTE: Uses the processor's current drop_list for all modes.
    For accurate mode-switch performance (Perf^S), call this BEFORE
    stable_hi_recovery modifies drop_list.
    """
    return {
        'L': normalized_performance(processors, all_lo_tasks, beta, 'L'),
        'S': normalized_performance(processors, all_lo_tasks, beta, 'S'),
        'H': normalized_performance(processors, all_lo_tasks, beta, 'H'),
    }
