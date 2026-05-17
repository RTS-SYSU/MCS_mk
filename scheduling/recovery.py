r"""
Stable-HI Recovery Algorithm (unified action-pool version).

When the system enters stable HI mode after the mode-switch interval:
  - Retained LO tasks (Gamma_LO*) remain active with x_h = x_s initially.
  - Suspended LO tasks (overline{Gamma}_LO*) may be recovered to baseline
    (m_i, k_i).

Recovery and augmentation are unified into a single candidate pool:
  - Recovery action:  recover a suspended task at baseline (m_i, k_i).
    Density: rho_base = mu_i / C_i^LO.  Gain: mu_i.
  - Augment action:   increase x_h of a retained task by 1 unit.
    Density: rho_aug  = beta * mu_i / ((k_i - m_i) * C_i^LO).
    Gain:   beta * mu_i / (k_i - m_i).

Each iteration, the highest-density feasible action with positive gain
is committed.  The process repeats until no feasible positive-gain
action remains.
"""

from dataclasses import dataclass
from typing import List, Optional

from core.task import Task
from scheduling.sched_test import calculate_wcrt_hi

# ---------------------------------------------------------------------------
# Sacrifice unit  (kept for MaxCount compatibility)
# ---------------------------------------------------------------------------

@dataclass
class _SacrificeUnit:
    """One unit of sacrifice: reduce x_h of a retained LO task by 1."""
    task: Task
    loss: float     # beta * mu_j / (k_j - m_j)
    density: float  # beta * mu_j / ((k_j - m_j) * C_LO_j)

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
# Unified action pool
# ---------------------------------------------------------------------------

@dataclass
class _Action:
    """A candidate action: recover a suspended task or augment a retained one."""
    task: Task
    kind: str       # 'recover' or 'augment'
    density: float  # gain per unit C_LO
    gain: float     # absolute importance gain


def _build_action_pool(
    tasks: List[Task],
    drop_list: List[Task],
    beta: float,
) -> List[_Action]:
    """Build the unified pool of recovery and augmentation actions."""
    actions: List[_Action] = []

    # Recovery actions: each suspended LO task
    for t in drop_list:
        if t.criticality != "LO":
            continue
        density = t.baseline_importance / t.wcet_lo
        gain = t.baseline_importance
        actions.append(_Action(task=t, kind='recover', density=density, gain=gain))

    # Augmentation actions: each remaining unit of x_h per retained LO task
    for t in tasks:
        if t.criticality != "LO" or t in drop_list:
            continue
        remaining = (t.mk.k - t.mk.m) - t.mk.x_h
        if remaining <= 0:
            continue
        unit_density = (beta * t.baseline_importance) / ((t.mk.k - t.mk.m) * t.wcet_lo)
        unit_gain = beta * t.baseline_importance / (t.mk.k - t.mk.m)
        for _ in range(remaining):
            actions.append(_Action(task=t, kind='augment',
                                   density=unit_density, gain=unit_gain))

    return actions


def _check_feasible(action: _Action,
                    tasks: List[Task],
                    drop_list: List[Task],
                    beta: float) -> tuple:
    """
    Check whether *action* is stable-HI schedulable and profitable.

    Returns (feasible: bool, q_min: int, sequence: list).
    For augment: q_min = 0, sequence = [].
    For recover: q_min >= 0 is the minimum sacrifice prefix length.
    """
    if action.kind == 'recover':
        return _check_recover_with_sacrifice(action.task, tasks, drop_list, beta)
    else:
        ok = _check_augment(action.task, tasks, drop_list)
        return (ok, 0, [])


def _check_recover_with_sacrifice(
    candidate: Task,
    tasks: List[Task],
    drop_list: List[Task],
    beta: float,
) -> tuple:
    """
    Check whether recovering *candidate* is feasible, possibly with sacrifice.

    Returns (feasible, q_min, sequence).
    q_min is the minimum number of sacrifice units needed.
    """
    sequence = _build_sacrifice_sequence(candidate, tasks, drop_list, beta)

    # First: try q = 0 (no sacrifice)
    if not sequence:
        if _feasible_recovery(candidate, tasks, drop_list, [], 0):
            return (True, 0, [])
        return (False, 0, [])

    # Check profitable bound
    q_max = _profitable_upper_bound(sequence, candidate.baseline_importance)
    if q_max < 0:
        return (False, 0, [])

    # Binary search for minimum feasible prefix
    q_min = _binary_search_feasible(candidate, tasks, drop_list, sequence, q_max)
    if q_min is None:
        return (False, 0, [])

    loss = sum(u.loss for u in sequence[:q_min])
    gain = candidate.baseline_importance - loss
    if gain <= 0:
        return (False, 0, [])

    return (True, q_min, sequence)


def _check_augment(task: Task,
                   tasks: List[Task],
                   drop_list: List[Task]) -> bool:
    """Check whether increasing x_h of *task* by 1 is HI-schedulable."""
    saved_xh = task.mk.x_h
    task.mk.x_h += 1

    all_ok = True
    for t in tasks:
        if t in drop_list:
            continue
        if t.priority > task.priority:
            _, ok = calculate_wcrt_hi(t, tasks, drop_list)
            if not ok:
                all_ok = False
                break

    task.mk.x_h = saved_xh
    return all_ok


def _commit_action(action: _Action,
                   tasks: List[Task],
                   drop_list: List[Task],
                   sequence: List[_SacrificeUnit] = None,
                   q_min: int = 0) -> None:
    """Permanently apply *action*. Modifies state in place."""
    if action.kind == 'recover':
        if sequence and q_min > 0:
            _commit_recovery(action.task, tasks, drop_list, sequence, q_min)
        else:
            action.task.mk.x_h = 0
            drop_list.remove(action.task)
    else:
        action.task.mk.x_h += 1



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
    Unified action-pool Stable-HI Recovery algorithm.

    Recovery and augmentation actions compete in a single pool, ordered
    by density (gain per unit C_LO).  Each iteration, the highest-density
    feasible action with positive gain is committed.  Repeats until no
    feasible positive-gain action remains.

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

    while True:
        actions = _build_action_pool(tasks, drop_list, beta)
        if not actions:
            break

        # Sort by density descending (highest gain per interference first)
        actions.sort(key=lambda a: a.density, reverse=True)

        # Find the first (highest-density) feasible action
        chosen = None
        best_q = 0
        best_seq: List[_SacrificeUnit] = []
        for action in actions:
            ok, q, seq = _check_feasible(action, tasks, drop_list, beta)
            if ok:
                chosen = action
                best_q = q
                best_seq = seq
                break

        if chosen is None:
            break

        _commit_action(chosen, tasks, drop_list, best_seq, best_q)
        if chosen.kind == 'recover':
            recovered.append(chosen.task)

    return recovered
