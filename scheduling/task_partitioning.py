r"""
Task-to-processor partitioning via Worst-Fit Decreasing (WFD).

Two-phase approach:
  Phase 1 — WFD by utilization only (no RTA).
    - HI tasks: sorted by HI utilization descending, placed to core with lowest HI util.
    - LO tasks: sorted by effective utilization ((m/k)*C_LO/T) descending,
      placed to core with lowest LO util.
  Phase 2 — Priority assignment, RTA validation, and classification.
    - HI tasks must pass LO-mode AND mode-switch RTA  → hard failure otherwise.
    - LO tasks must pass LO-mode RTA                     → hard failure otherwise.
    - LO tasks that pass LO mode: mode-switch pass → retained (Gamma_LO*),
      mode-switch fail → suspended (overline{Gamma}_LO*).
"""

from typing import List, Optional

from core.processor import Processor
from core.task import Task
from scheduling.priority_assignment import assign_static_priorities
from scheduling.sched_test import (
    test_aMC,
    calculate_wcrt_lo,
)


def partition_tasks(original_tasks: List[Task],
                    num_processors: int) -> Optional[List[Processor]]:
    """
    Partition tasks and classify LO tasks per core.

    Returns list of Processors on success, None on failure.
    """
    processors = [Processor(i) for i in range(num_processors)]

    hi_tasks = [t for t in original_tasks if t.criticality == "HI"]
    lo_tasks = [t for t in original_tasks if t.criticality == "LO"]

    # ==================================================================
    # Phase 1 — WFD by utilization (no RTA), baseline mk (all x = 0)
    # ==================================================================

    # HI tasks: descending HI utilization
    hi_tasks.sort(key=lambda t: t.wcet_hi / t.period, reverse=True)

    for task in hi_tasks:
        processors.sort(key=lambda p: p.utilization_hi)  # worst-fit
        placed = False
        for p in processors:
            u_lo = p.utilization_lo + task.wcet_lo / task.period
            u_hi = p.utilization_hi + task.wcet_hi / task.period
            if u_lo <= 1.0 and u_hi <= 1.0:
                p.add_task(task)
                placed = True
                break
        if not placed:
            return None

    # LO tasks: descending effective utilization
    lo_tasks.sort(key=lambda t: (t.mk.m / t.mk.k) * (t.wcet_lo / t.period),
                  reverse=True)

    for task in lo_tasks:
        processors.sort(key=lambda p: p.utilization_lo)  # worst-fit
        placed = False
        eff = task.mk.m / task.mk.k
        u_inc = (task.wcet_lo / task.period) * eff
        for p in processors:
            if p.utilization_lo + u_inc <= 1.0:
                p.add_task(task)
                placed = True
                break
        if not placed:
            return None

    # ==================================================================
    # Phase 2 — Priority assignment, RTA validation, classification
    #
    # All tasks on each core are processed in priority order (highest
    # first).  HI and LO tasks are interleaved: a high-priority LO task
    # that is suspended must be excluded from lower-priority HI / LO
    # tasks' mode-switch interference.
    # ==================================================================

    for p in processors:
        assign_static_priorities(p.tasks)
        active_drops: List[Task] = []

        # Process in priority order (highest → lowest)
        for task in sorted(p.tasks, key=lambda t: t.priority, reverse=True):
            if task.criticality == "HI":
                R_lo, ok_lo = calculate_wcrt_lo(task, p.tasks)
                if not ok_lo:
                    return None
                if not test_aMC(task, p.tasks, R_lo, drop_tasks=active_drops):
                    return None
            else:  # LO
                R_lo, ok_lo = calculate_wcrt_lo(task, p.tasks)
                if not ok_lo:
                    return None  # hard fail: must pass LO mode
                if not test_aMC(task, p.tasks, R_lo, drop_tasks=active_drops):
                    active_drops.append(task)
                    p.mark_as_dropped(task)

    return processors
