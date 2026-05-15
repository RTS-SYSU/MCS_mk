r"""
Stable-HI Recovery Algorithm.

When the system enters stable HI mode after the mode-switch interval:
  - Retained LO tasks (Gamma_LO*) remain active with x_h = x_s initially.
  - Suspended LO tasks (overline{Gamma}_LO*) may be recovered to baseline
    (m_i, k_i) by sacrificing augmented x from retained tasks.

The algorithm iteratively selects the suspended task with the highest net
importance gain, sacrifices the minimum set of augmented units (ordered by
ascending augmented density, limited to higher-priority retained tasks),
and checks feasibility via stable-HI RTA.

Only recoveries with positive net gain (gain = mu_i - sacrifice_loss > 0)
are accepted.
"""

from dataclasses import dataclass
from typing import List, Optional

from core.task import Task
from scheduling.sched_test import calculate_wcrt_hi


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _augmented_density(task: Task, beta: float) -> float:
    """rho_i^aug = (beta * mu_i) / ((k_i - m_i) * C_i^LO)"""
    denom = (task.mk.k - task.mk.m) * task.wcet_lo
    if denom <= 0:
        return 0.0
    return (beta * task.baseline_importance) / denom


def _all_schedulable_hi(tasks: List[Task], drop_list: List[Task]) -> bool:
    """Check whether every active task passes stable-HI RTA."""
    for t in tasks:
        if t in drop_list:
            continue
        _, ok = calculate_wcrt_hi(t, tasks, drop_list)
        if not ok:
            return False
    return True


# ---------------------------------------------------------------------------
# Sacrifice unit
# ---------------------------------------------------------------------------

@dataclass
class _SacrificeUnit:
    """One unit of sacrifice: reduce x_h of a retained LO task by 1."""
    task: Task
    loss: float     # beta * mu_j / (k_j - m_j)
    density: float  # beta * mu_j / ((k_j - m_j) * C_LO_j)


# ---------------------------------------------------------------------------
# Build effective sacrifice sequence for a candidate
# ---------------------------------------------------------------------------

def _build_sacrifice_sequence(
    candidate: Task,
    tasks: List[Task],
    drop_list: List[Task],
    beta: float,
) -> List[_SacrificeUnit]:
    """
    Build sacrifice units from retained LO tasks that are:
      - have x_h > 0
      - have k > m  (augmentable)
      - NOT in drop_list (i.e. retained / Gamma_LO*)

    Units are split into two groups:
      - hi_pri:  priority <  candidate.priority  (effective — reduces
                 interference for the candidate)
      - lo_pri:  priority >  candidate.priority  (backup — frees system
                 utilisation but does not directly help the candidate)

    Each group is sorted internally by augmented density ascending.
    The final sequence is hi_pri units followed by lo_pri units.
    """
    def _make_units(t: Task) -> List[_SacrificeUnit]:
        loss_per_unit = beta * t.baseline_importance / (t.mk.k - t.mk.m)
        density = loss_per_unit / t.wcet_lo
        return [_SacrificeUnit(task=t, loss=loss_per_unit, density=density)
                for _ in range(t.mk.x_h)]

    hi_units: List[_SacrificeUnit] = []
    lo_units: List[_SacrificeUnit] = []

    for t in tasks:
        if t.criticality != "LO":
            continue
        if t in drop_list:
            continue
        if t.mk.k == t.mk.m or t.mk.x_h <= 0:
            continue

        if t.priority < candidate.priority:
            hi_units.extend(_make_units(t))
        elif t.priority > candidate.priority:
            lo_units.extend(_make_units(t))

    hi_units.sort(key=lambda u: u.density)
    lo_units.sort(key=lambda u: u.density)

    return hi_units + lo_units


# ---------------------------------------------------------------------------
# Profitable upper bound
# ---------------------------------------------------------------------------

def _profitable_upper_bound(
    sequence: List[_SacrificeUnit],
    candidate_mu: float,
) -> int:
    """
    Find the largest prefix length q such that total sacrifice loss is still
    strictly less than the candidate's baseline importance.

    Returns q_max, or -1 if even q=0 is unprofitable (should not happen).
    """
    total_loss = 0.0
    q_max = -1
    for q, unit in enumerate(sequence):
        total_loss += unit.loss
        if total_loss < candidate_mu:
            q_max = q + 1  # q is 0-indexed; prefix length = q+1
        else:
            break
    return q_max


# ---------------------------------------------------------------------------
# Feasibility oracle for a given sacrifice prefix
# ---------------------------------------------------------------------------

def _feasible_recovery(
    candidate: Task,
    tasks: List[Task],
    drop_list: List[Task],
    sequence: List[_SacrificeUnit],
    q: int,
) -> bool:
    """
    Check whether recovering *candidate* with the first *q* sacrifice units
    applied is stable-HI schedulable.

    Steps:
      1. Save and apply sacrifice (reduce x_h of affected retained tasks).
      2. Temporarily recover candidate (remove from drop_list, x_h = 0).
      3. Run HI-mode RTA on candidate and all lower-priority active tasks.
      4. Restore original x_h values.

    Returns True only if ALL affected tasks pass HI-mode RTA.
    """
    # --- save current x_h ---
    saved: dict[int, int] = {}
    for t in tasks:
        if t.criticality == "LO":
            saved[t.id] = t.mk.x_h

    # --- apply sacrifice prefix ---
    task_by_id = {t.id: t for t in tasks}
    unit_by_task: dict[int, int] = {}
    for unit in sequence[:q]:
        unit_by_task[unit.task.id] = unit_by_task.get(unit.task.id, 0) + 1
    for tid, count in unit_by_task.items():
        task_by_id[tid].mk.x_h = max(0, task_by_id[tid].mk.x_h - count)

    # --- temporarily recover candidate ---
    temp_drops = [t for t in drop_list if t.id != candidate.id]
    saved_cand_xh = candidate.mk.x_h
    candidate.mk.x_h = 0

    # --- affected: candidate + lower-priority active tasks ---
    affected = [candidate]
    for t in tasks:
        if t in temp_drops:
            continue
        if t.priority > candidate.priority:
            affected.append(t)

    # --- run HI-mode RTA ---
    all_ok = True
    for t in affected:
        _, ok = calculate_wcrt_hi(t, tasks, temp_drops)
        if not ok:
            all_ok = False
            break

    # --- restore ---
    for t in tasks:
        if t.criticality == "LO" and t.id in saved:
            t.mk.x_h = saved[t.id]
    candidate.mk.x_h = saved_cand_xh

    return all_ok


# ---------------------------------------------------------------------------
# Binary search for minimum feasible prefix
# ---------------------------------------------------------------------------

def _binary_search_feasible(
    candidate: Task,
    tasks: List[Task],
    drop_list: List[Task],
    sequence: List[_SacrificeUnit],
    q_max: int,
) -> Optional[int]:
    """
    Binary search in [0, q_max] for the smallest q such that the recovery
    is feasible.  Returns q_min, or None if even q_max is infeasible.
    """
    # Quick check: is recovery feasible at max sacrifice?
    if not _feasible_recovery(candidate, tasks, drop_list, sequence, q_max):
        return None

    lo, hi = 0, q_max
    best = q_max

    while lo <= hi:
        mid = (lo + hi) // 2
        if _feasible_recovery(candidate, tasks, drop_list, sequence, mid):
            best = mid
            hi = mid - 1  # try less sacrifice
        else:
            lo = mid + 1  # need more sacrifice

    return best


# ---------------------------------------------------------------------------
# Commit: permanently apply a recovery
# ---------------------------------------------------------------------------

def _commit_recovery(
    candidate: Task,
    tasks: List[Task],
    drop_list: List[Task],
    sequence: List[_SacrificeUnit],
    q: int,
) -> None:
    """Apply sacrifice and recover candidate. Modifies tasks and drop_list in place."""
    # Apply sacrifice
    task_by_id = {t.id: t for t in tasks}
    unit_by_task: dict[int, int] = {}
    for unit in sequence[:q]:
        unit_by_task[unit.task.id] = unit_by_task.get(unit.task.id, 0) + 1
    for tid, count in unit_by_task.items():
        task_by_id[tid].mk.x_h = max(0, task_by_id[tid].mk.x_h - count)

    # Recover candidate
    candidate.mk.x_h = 0
    drop_list.remove(candidate)


# ---------------------------------------------------------------------------
# Post-recovery HI-mode augmentation
# ---------------------------------------------------------------------------

def _hi_mode_augment(
    tasks: List[Task],
    drop_list: List[Task],
    beta: float,
) -> None:
    """
    After recovery, try to increase x_h for retained LO tasks using any
    remaining HI-mode slack.

    Tasks are processed in descending augmented importance density order
    (highest benefit first).  For each task, search for the maximum x_h
    that keeps all active tasks HI-schedulable.
    """
    eligible = [t for t in tasks
                if t.criticality == "LO"
                and t not in drop_list
                and t.mk.k > t.mk.m]

    if not eligible:
        return

    eligible.sort(key=lambda t: _augmented_density(t, beta), reverse=True)

    for task in eligible:
        max_x = task.mk.k - task.mk.m
        best_x = task.mk.x_h  # start from current x_h, try to go higher

        for x in range(max_x, best_x, -1):
            task.mk.set_x(x, 'H')
            if _all_schedulable_hi(tasks, drop_list):
                best_x = x
                break

        task.mk.set_x(best_x, 'H')


# ---------------------------------------------------------------------------
# Main Stable-HI Recovery flow
# ---------------------------------------------------------------------------

def stable_hi_recovery(
    tasks: List[Task],
    drop_list: List[Task],
    beta: float = 0.5,
) -> List[Task]:
    """
    Main Stable-HI Recovery algorithm.

    1. Sort suspended LO tasks by baseline importance density (mu_i / C_LO_i)
       descending.  Try to recover each in order: the highest-density task
       gets the first chance at the available augmented x_h pool.

    2. For each candidate, build an effective sacrifice sequence from
       higher-priority retained LO tasks.  Search for the minimum feasible
       sacrifice prefix within the profitable range.  If gain > 0, commit.

    3. After recovery, use any remaining HI-mode slack to increase x_h of
       retained LO tasks (descending augmented importance density).

    Args:
        tasks: All tasks on the core (must have priorities assigned).
        drop_list: Suspended LO tasks (overline{Gamma}_LO*).
        beta: Augmented importance gain parameter.

    Returns:
        The list of recovered tasks (in recovery order).
    """
    if drop_list is None:
        drop_list = []

    # Initialise x_h from x_s for retained, 0 for suspended
    for t in tasks:
        if t.criticality == "LO" and t not in drop_list:
            t.mk.x_h = t.mk.x_s
        elif t.criticality == "LO":
            t.mk.x_h = 0

    recovered: List[Task] = []

    # --- Recovery pass ---
    suspended = [t for t in drop_list if t.criticality == "LO"]
    suspended.sort(
        key=lambda t: t.baseline_importance / t.wcet_lo,
        reverse=True,
    )

    for candidate in suspended:
        if candidate not in drop_list:
            continue  # already recovered in an earlier iteration

        sequence = _build_sacrifice_sequence(candidate, tasks, drop_list, beta)

        # If no sacrifice units are available, check whether the candidate
        # can be recovered at baseline (q = 0) with current slack.
        if not sequence:
            if _feasible_recovery(candidate, tasks, drop_list, [], 0):
                _commit_recovery(candidate, tasks, drop_list, [], 0)
                recovered.append(candidate)
            continue

        q_max = _profitable_upper_bound(sequence, candidate.baseline_importance)
        if q_max < 0:
            continue

        q_min = _binary_search_feasible(candidate, tasks, drop_list, sequence, q_max)
        if q_min is None:
            continue

        loss = sum(u.loss for u in sequence[:q_min])
        gain = candidate.baseline_importance - loss
        if gain <= 0:
            continue

        _commit_recovery(candidate, tasks, drop_list, sequence, q_min)
        recovered.append(candidate)

    # --- Post-recovery augmentation ---
    _hi_mode_augment(tasks, drop_list, beta)

    return recovered
