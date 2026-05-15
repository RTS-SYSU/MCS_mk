r"""
Response-Time Analysis for Mixed-Criticality Systems with Weakly-Hard Constraints.

Three WCRT analyses:
  - R_i^{LO}:  LO-mode WCRT
  - R_i^*:     Mode-switch WCRT
  - R_i^{HI}:  Stable HI-mode WCRT

The mandatory-job upper bound eta(t, T, m, k) counts mandatory jobs in [0,t).
"""

import math
from typing import List, Optional, Tuple

from core.task import Task


# ---------------------------------------------------------------------------
# η(t, T, m, k) — mandatory jobs in an interval of length t
# ---------------------------------------------------------------------------

def calculate_mk_jobs(t: float, T: float, m: int, k: int) -> int:
    """
    Maximum number of mandatory jobs of a weakly-hard task in interval [0, t).

    η(t, T, m, k) = floor(ceil(t/T) / k) * m + min(ceil(t/T) mod k, m)
    """
    if t <= 0:
        return 0
    N = math.ceil(t / T)
    cycles = N // k
    remainder = N % k
    return cycles * m + min(remainder, m)


# ---------------------------------------------------------------------------
# R_i^{LO} — LO-mode WCRT
# ---------------------------------------------------------------------------

def calculate_wcrt_lo(cur_task: Task, tasks: List[Task],
                      max_iter: int = 1000, tol: float = 1e-6) -> Tuple[float, bool]:
    """
    R_i^{LO} = C_i^{LO}
        + Σ_{τ_h ∈ hpH_i} ⌈R_i^{LO}/T_h⌉ * C_h^{LO}
        + Σ_{τ_l ∈ hpL_i} η(R_i^{LO}, T_l, m_l + x_l^L, k_l) * C_l^{LO}
    """
    C_lo = cur_task.wcet_lo
    hp_tasks = [t for t in tasks if t.priority < cur_task.priority]
    hp_hi = [t for t in hp_tasks if t.criticality == "HI"]
    hp_lo = [t for t in hp_tasks if t.criticality == "LO"]

    R_prev = 0.0
    R = C_lo
    iter_count = 0

    while abs(R - R_prev) > tol and iter_count <= max_iter:
        R_prev = R

        interference_hi = sum(math.ceil(R_prev / t.period) * t.wcet_lo for t in hp_hi)

        interference_lo = 0
        for t in hp_lo:
            m_eff = t.mk.m + t.mk.x_l  # m_l + x_l^L
            num_mand = calculate_mk_jobs(R_prev, t.period, m_eff, t.mk.k)
            interference_lo += num_mand * t.wcet_lo

        R = C_lo + interference_hi + interference_lo

        if R > cur_task.deadline:
            return R, False
        iter_count += 1

    if iter_count > max_iter:
        return R, False
    return R, R <= cur_task.deadline


# ---------------------------------------------------------------------------
# R_i^* — Mode-switch WCRT
# ---------------------------------------------------------------------------

def calculate_wcrt_mc(cur_task: Task, tasks: List[Task], R_lo: float,
                      drop_tasks: List[Task] = None,
                      max_iter: int = 1000, tol: float = 1e-6) -> Tuple[float, bool]:
    r"""
    R_i^* = C_i^{L_i}
        + sum_{h in hpH_i} ceil(R_i^*/T_h) * C_h^{HI}
        + sum_{l in hpL_i} eta(R_i^{LO}, T_l, m_l + x_l^L, k_l) * C_l^{LO}
        + sum_{l in hpL_i \ drop} [eta(R_i^*, T_l, m_l + x_l^S, k_l)
                                     - eta(R_i^{LO}, T_l, m_l + x_l^S, k_l)] * C_l^{LO}

    C_i^{L_i} = wcet_hi for HI, wcet_lo for LO.
    Dropped LO tasks are excluded from the continuation part.
    """
    if drop_tasks is None:
        drop_tasks = []

    if cur_task.criticality == "HI":
        C_max = cur_task.wcet_hi
    else:
        if cur_task in drop_tasks:
            return 0.0, True  # dropped LO task has no mode-switch requirement
        C_max = cur_task.wcet_lo

    hp_tasks = [t for t in tasks if t.priority < cur_task.priority]
    hp_hi = [t for t in hp_tasks if t.criticality == "HI"]
    hp_lo = [t for t in hp_tasks if t.criticality == "LO"]

    # Pre-compute the LO-mode contribution (uses x_l^L for all LO hp tasks)
    lo_contribution_fixed = 0.0
    for t in hp_lo:
        m_eff_L = t.mk.m + t.mk.x_l
        lo_contribution_fixed += calculate_mk_jobs(R_lo, t.period, m_eff_L, t.mk.k) * t.wcet_lo

    R_prev = 0.0
    R = R_lo  # initial guess
    iter_count = 0

    while abs(R - R_prev) > tol and iter_count < max_iter:
        R_prev = R

        interference_hi = sum(math.ceil(R_prev / t.period) * t.wcet_hi for t in hp_hi)

        # Continuation part: only non-dropped LO tasks with x_l^S
        lo_continuation = 0.0
        for t in hp_lo:
            if t in drop_tasks:
                continue
            m_eff_S = t.mk.m + t.mk.x_s
            total_m = calculate_mk_jobs(R_prev, t.period, m_eff_S, t.mk.k)
            pre_m = calculate_mk_jobs(R_lo, t.period, m_eff_S, t.mk.k)
            lo_continuation += max(0, total_m - pre_m) * t.wcet_lo

        R = C_max + interference_hi + lo_contribution_fixed + lo_continuation

        if R > cur_task.deadline:
            return R, False
        iter_count += 1

    if iter_count >= max_iter:
        return R, False
    return R, R <= cur_task.deadline


# ---------------------------------------------------------------------------
# R_i^{HI} — Stable HI-mode WCRT
# ---------------------------------------------------------------------------

def calculate_wcrt_hi(cur_task: Task, tasks: List[Task],
                      drop_tasks: List[Task] = None,
                      max_iter: int = 1000, tol: float = 1e-6) -> Tuple[float, bool]:
    r"""
    R_i^{HI} = C_i^{L_i}
        + sum_{h in hpH_i} ceil(R_i^{HI}/T_h) * C_h^{HI}
        + sum_{l in hpL_i \ drop} eta(R_i^{HI}, T_l, m_l + x_l^H, k_l) * C_l^{LO}
    """
    if drop_tasks is None:
        drop_tasks = []

    if cur_task.criticality == "HI":
        C_max = cur_task.wcet_hi
    else:
        if cur_task in drop_tasks:
            return 0.0, True  # dropped LO task is suspended
        C_max = cur_task.wcet_lo

    hp_tasks = [t for t in tasks if t.priority < cur_task.priority]
    hp_hi = [t for t in hp_tasks if t.criticality == "HI"]
    hp_lo = [t for t in hp_tasks if t.criticality == "LO" and t not in drop_tasks]

    R_prev = 0.0
    R = C_max
    iter_count = 0

    while abs(R - R_prev) > tol and iter_count <= max_iter:
        R_prev = R

        interference_hi = sum(math.ceil(R_prev / t.period) * t.wcet_hi for t in hp_hi)

        interference_lo = 0
        for t in hp_lo:
            m_eff_H = t.mk.m + t.mk.x_h  # m_l + x_l^H
            num_mand = calculate_mk_jobs(R_prev, t.period, m_eff_H, t.mk.k)
            interference_lo += num_mand * t.wcet_lo

        R = C_max + interference_hi + interference_lo

        if R > cur_task.deadline:
            return R, False
        iter_count += 1

    if iter_count > max_iter:
        return R, False
    return R, R <= cur_task.deadline


# ---------------------------------------------------------------------------
# Individual schedulability tests
# ---------------------------------------------------------------------------

def test_aLO(task: Task, tasks: List[Task]) -> bool:
    """Test whether a single task passes LO-mode RTA."""
    _, ok = calculate_wcrt_lo(task, tasks)
    return ok


def test_aMC(task: Task, tasks: List[Task], R_lo: float,
             drop_tasks: List[Task] = None) -> bool:
    """Test whether a single task passes mode-switch RTA."""
    _, ok = calculate_wcrt_mc(task, tasks, R_lo, drop_tasks)
    return ok


# ---------------------------------------------------------------------------
# Full task-set schedulability tests
# ---------------------------------------------------------------------------

def schedulability_test_lo(tasks: List[Task]) -> bool:
    """
    Test LO-mode schedulability for all tasks in the set.
    """
    for task in tasks:
        _, ok = calculate_wcrt_lo(task, tasks)
        if not ok:
            return False
    return True


def schedulability_test_hi(tasks: List[Task],
                           drop_tasks: List[Task] = None) -> bool:
    """
    Full MCS schedulability test: LO mode → mode switch → HI mode for every task.
    Returns True only if all three analyses pass for each task.
    """
    if drop_tasks is None:
        drop_tasks = []

    for task in tasks:
        # LO mode
        R_lo, ok_lo = calculate_wcrt_lo(task, tasks)
        if not ok_lo:
            return False

        # Mode switch
        _, ok_mc = calculate_wcrt_mc(task, tasks, R_lo, drop_tasks)
        if not ok_mc:
            return False

        # Stable HI mode
        _, ok_hi = calculate_wcrt_hi(task, tasks, drop_tasks)
        if not ok_hi:
            return False

    return True


def classify_lo_tasks(tasks: List[Task],
                      drop_tasks: List[Task] = None) -> Optional[Tuple[List[Task], List[Task]]]:
    r"""
    Classify LO tasks into retained (Gamma_LO*) and suspended (overline{Gamma}_LO*)
    based on mode-switch schedulability.

    LO tasks are processed in priority order (highest first). When a task is
    classified as suspended it is excluded from subsequent tasks' mode-switch
    interference.

    LO-mode failure is a hard error: returns None.
    Mode-switch failure -> suspended.

    Returns: (retained, suspended) on success, None if any LO task fails LO-mode RTA.
    """
    if drop_tasks is None:
        drop_tasks = []

    # Work on a copy so we can accumulate suspended tasks without mutating
    # the caller's list until we're done.
    active_drops = list(drop_tasks)

    # Only LO tasks, sorted by priority: highest priority first (smallest number)
    lo_tasks = sorted(
        [t for t in tasks if t.criticality == "LO"],
        key=lambda t: t.priority,
    )

    retained: List[Task] = []
    suspended: List[Task] = []

    for task in lo_tasks:
        R_lo, ok_lo = calculate_wcrt_lo(task, tasks)
        if not ok_lo:
            return None  # hard failure

        _, ok_mc = calculate_wcrt_mc(task, tasks, R_lo, active_drops)
        if ok_mc:
            retained.append(task)
        else:
            suspended.append(task)
            active_drops.append(task)

    return retained, suspended
