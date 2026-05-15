r"""
LO-Mode Augmentation and Mode-Switch Degradation.

LO-Mode Augmentation:
  In LO mode all LO tasks are active (a_i^L = 1), but only tasks in Gamma_LO*
  are eligible for augmentation.  Tasks in overline{Gamma}_LO* keep baseline
  (x_i^L = 0).

  Promotion order is determined by augmented importance density:
      rho_i^aug = (beta * mu_i) / ((k_i - m_i) * C_i^LO)

  For each eligible task:
      x_i^L = max { x | 0 <= x <= k_i - m_i,  all tasks LO-schedulable }

Mode-Switch Degradation:
  When the system switches to HI mode, HI tasks execute with C^HI.  The LO-mode
  augmented levels x_i^L may no longer be schedulable.  Each retained LO task
  degrades its augmented level:
      x_i^L -> x_i^S,    0 <= x_i^S <= x_i^L

  Only Gamma_LO* tasks remain active during mode switch; overline{Gamma}_LO*
  tasks are suspended.  Retained tasks never lose baseline (m_i, k_i).

  Degradation order prioritizes tasks with lower augmented benefit (ascending
  density), so low-benefit tasks are degraded first, preserving high-benefit
  tasks' augmented levels.
      x_i^S = max { x | 0 <= x <= x_i^L,  all tasks MC-schedulable }
"""

from typing import List

from core.task import Task, Mode
from scheduling.sched_test import calculate_wcrt_lo, calculate_wcrt_mc


def _augmented_density(task: Task, beta: float) -> float:
    """rho_i^aug = (beta * mu_i) / ((k_i - m_i) * C_i^LO)"""
    denom = (task.mk.k - task.mk.m) * task.wcet_lo
    if denom <= 0:
        return 0.0
    return (beta * task.baseline_importance) / denom


def _all_schedulable_lo(tasks: List[Task]) -> bool:
    """Check whether every task passes LO-mode RTA."""
    for t in tasks:
        _, ok = calculate_wcrt_lo(t, tasks)
        if not ok:
            return False
    return True


def _all_schedulable_mc(tasks: List[Task], drop_list: List[Task]) -> bool:
    """Check whether every active task passes mode-switch RTA."""
    for t in tasks:
        if t in drop_list:
            continue
        R_lo, ok_lo = calculate_wcrt_lo(t, tasks)
        if not ok_lo:
            return False
        _, ok_mc = calculate_wcrt_mc(t, tasks, R_lo, drop_list)
        if not ok_mc:
            return False
    return True


# ---------------------------------------------------------------------------
# LO-Mode Augmentation
# ---------------------------------------------------------------------------

def lo_mode_augment(tasks: List[Task],
                    drop_list: List[Task] = None,
                    beta: float = 0.5) -> None:
    """
    Augment x_i^L for eligible LO tasks (those NOT in drop_list)
    using density-ordered, schedulability-aware search.

    Modifies task.mk.x_l in place. Tasks with k_i == m_i are skipped.

    Args:
        tasks: All tasks on a core (must have priorities assigned).
        drop_list: Tasks suspended in mode switch (overline{Gamma}_LO*).
                   These remain active in LO mode but keep x_i^L = 0.
        beta: Augmented importance gain parameter.
    """
    if drop_list is None:
        drop_list = []

    for t in tasks:
        if t.criticality == "LO":
            t.mk.reset_x('L')

    eligible = [t for t in tasks
                if t.criticality == "LO"
                and t not in drop_list
                and t.mk.k > t.mk.m]

    if not eligible:
        return

    eligible.sort(key=lambda t: _augmented_density(t, beta), reverse=True)

    for task in eligible:
        max_x = task.mk.k - task.mk.m

        lo, hi = 0, max_x
        best_x = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            task.mk.set_x(mid, 'L')
            if _all_schedulable_lo(tasks):
                best_x = mid
                lo = mid + 1
            else:
                hi = mid - 1

        task.mk.set_x(best_x, 'L')


# ---------------------------------------------------------------------------
# Mode-Switch Degradation
# ---------------------------------------------------------------------------

def mode_switch_degrade(tasks: List[Task],
                        drop_list: List[Task] = None,
                        beta: float = 0.5) -> None:
    """
    Degrade x_i^L -> x_i^S for retained LO tasks to maintain mode-switch
    schedulability.

    Only tasks NOT in drop_list (Gamma_LO*) are considered; suspended tasks
    (overline{Gamma}_LO*) are already excluded from mode-switch analysis.

    Degradation order: ascending augmented density (lowest benefit first).

    Modifies task.mk.x_s in place.

    Args:
        tasks: All tasks on a core (must have priorities assigned).
        drop_list: Suspended LO tasks (overline{Gamma}_LO*).
        beta: Augmented importance gain parameter.
    """
    if drop_list is None:
        drop_list = []

    # Start from LO-mode augmented level
    for t in tasks:
        if t.criticality == "LO":
            t.mk.x_s = t.mk.x_l  # initialise x_s from x_l

    # Only retained LO tasks with k_i > m_i are eligible for degradation
    retained = [t for t in tasks
                if t.criticality == "LO"
                and t not in drop_list
                and t.mk.k > t.mk.m]

    if not retained:
        return

    # Sort by density ascending: lower benefit -> degraded first
    retained.sort(key=lambda t: _augmented_density(t, beta))

    for task in retained:
        max_x = task.mk.x_s

        lo, hi = 0, max_x
        best_x = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            task.mk.set_x(mid, 'S')
            if _all_schedulable_mc(tasks, drop_list):
                best_x = mid
                lo = mid + 1
            else:
                hi = mid - 1

        task.mk.set_x(best_x, 'S')

