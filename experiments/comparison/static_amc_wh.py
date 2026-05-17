r"""
Static-AMC-WH baseline.

LO mode:  x_l = k - m  (full augmentation) for all LO tasks.
MC mode:  x_s = 0       (baseline only).  Drop tasks that fail MC RTA.
HI mode:  x_h = 0       (baseline only).  No recovery, no augmentation.

If a mode is not schedulable for all required tasks, Perf^mode = 0.
"""

from typing import Dict, List, Tuple

from core.processor import Processor
from core.task import Task
from scheduling.sched_test import calculate_wcrt_lo, calculate_wcrt_mc, calculate_wcrt_hi
from experiments.performance import global_max_importance, core_importance


def run(processors: List[Processor],
        all_lo_tasks: List[Task],
        beta: float = 0.5) -> Dict[str, float]:
    """
    Apply Static-AMC-WH and return {L: Perf^L, S: Perf^S, H: Perf^H}.

    If a mode is infeasible, its performance is 0.
    """
    # --- 1. Set x values ---
    for p in processors:
        for t in p.tasks:
            if t.criticality != "LO":
                continue
            t.mk.set_x(t.mk.k - t.mk.m, 'L')  # max in LO
            t.mk.set_x(0, 'S')                  # baseline in MC
            t.mk.set_x(0, 'H')                  # baseline in HI

    # --- 2. LO mode: all LO tasks active, must be schedulable ---
    lo_ok = True
    for p in processors:
        for t in p.tasks:
            _, ok = calculate_wcrt_lo(t, p.tasks)
            if not ok:
                lo_ok = False
                break
        if not lo_ok:
            break

    i_max = global_max_importance(all_lo_tasks, beta)
    if i_max == 0:
        return {'L': 0.0, 'S': 0.0, 'H': 0.0}

    perf_l = 0.0
    if lo_ok:
        perf_l = sum(core_importance(p.tasks, [], beta, 'L')
                     for p in processors) / i_max

    # --- 3. MC mode: classify + drop ---
    # Need to do classification: for each LO task, check if it passes MC RTA.
    # HI tasks must also pass MC.
    for p in processors:
        p.drop_list.clear()

    mc_ok = True
    for p in processors:
        # Classify LO tasks
        for t in p.tasks:
            if t.criticality != "LO":
                continue
            R_lo, ok_lo = calculate_wcrt_lo(t, p.tasks)
            if not ok_lo:
                p.mark_as_dropped(t)
                continue
            _, ok_mc = calculate_wcrt_mc(t, p.tasks, R_lo, p.drop_list)
            if not ok_mc:
                p.mark_as_dropped(t)

        # Verify HI tasks pass MC (with accumulated drops)
        for t in p.tasks:
            if t.criticality != "HI":
                continue
            R_lo, ok_lo = calculate_wcrt_lo(t, p.tasks)
            if not ok_lo:
                mc_ok = False
                break
            _, ok_mc = calculate_wcrt_mc(t, p.tasks, R_lo, p.drop_list)
            if not ok_mc:
                mc_ok = False
                break
        if not mc_ok:
            break

    drops_mc = [list(p.drop_list) for p in processors]

    perf_s = 0.0
    if mc_ok:
        perf_s = sum(core_importance(p.tasks, drops_mc[p.id], beta, 'S')
                     for p in processors) / i_max

    # --- 4. HI mode: same drops as MC, x_h = 0, no recovery ---
    hi_ok = True
    for p in processors:
        for t in p.tasks:
            if t in p.drop_list:
                continue
            _, ok = calculate_wcrt_hi(t, p.tasks, p.drop_list)
            if not ok:
                hi_ok = False
                break
        if not hi_ok:
            break

    perf_h = 0.0
    if hi_ok:
        perf_h = sum(core_importance(p.tasks, drops_mc[p.id], beta, 'H')
                     for p in processors) / i_max

    return {'L': perf_l, 'S': perf_s, 'H': perf_h}
