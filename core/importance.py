"""
Augmented Weakly-Hard Importance Model.

For each LO task τ_i with baseline (m_i, k_i) and augmented level x_i:
    I_i(x_i) = μ_i * (1 + β * x_i / (k_i - m_i)),   0 < β < 1

Total LO-task importance under mode χ ∈ {L, S, H}:
    I^χ = Σ a_i^χ * I_i(x_i^χ)
where a_i^χ ∈ {0,1} indicates whether τ_i is active in mode χ,
and x_i^χ is the augmented level in that mode.
"""

from typing import List

from core.task import Task, Mode

# Global β: controls the relative gain of the augmented part.
BETA: float = 0.5


def set_beta(beta: float):
    global BETA
    if not (0.0 < beta < 1.0):
        raise ValueError(f"beta must be in (0, 1), got {beta}")
    BETA = beta


def task_importance(task: Task, beta: float = None, mode: Mode = 'L') -> float:
    """
    Return I_i(x_i^χ) for a single task at the given mode's augmented level.
    For inactive LO tasks the caller should return 0 instead.
    """
    if beta is None:
        beta = BETA
    return task.calculate_importance(beta, mode)


def total_importance(tasks: List[Task],
                     active_flags: List[bool] = None,
                     beta: float = None,
                     mode: Mode = 'L') -> float:
    """
    Compute total LO-task importance I^χ = Σ a_i^χ * I_i(x_i^χ).

    Args:
        tasks: All tasks in the system.
        active_flags: Per-task a_i^χ flags (True = active). If None, all LO tasks
                      are assumed active and HI tasks are excluded.
        beta: Augmented gain parameter (uses global BETA if None).
        mode: 'L' (x^L), 'S' (x^S), 'H' (x^H).
    """
    if beta is None:
        beta = BETA
    total = 0.0
    for i, task in enumerate(tasks):
        if task.criticality == "HI":
            continue
        active = active_flags[i] if active_flags is not None else True
        if active:
            total += task.calculate_importance(beta, mode)
    return total
