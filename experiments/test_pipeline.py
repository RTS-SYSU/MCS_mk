r"""
End-to-end pipeline test: generation → partition → classify →
LO augment → MC degrade → HI recovery → performance.

Outputs per-core task parameters, drop/recovery status, x values,
and normalized performance for each mode.
"""

import random
from typing import List

from core.task import Task
from core.processor import Processor
from core.importance import BETA, set_beta
from scheduling.priority_assignment import assign_static_priorities
from scheduling.task_partitioning import partition_tasks
from scheduling.augmentation import lo_mode_augment, mode_switch_degrade
from scheduling.recovery import stable_hi_recovery
from experiments.performance import (
    normalized_performance,
    core_importance,
    global_max_importance,
)

from utils.generate_taskset import generate_taskset


def run_pipeline(
    num_processors: int = 4,
    total_tasks: int = 20,
    target_util: float = 0.6,
    cp: float = 0.5,       # fraction of HI tasks
    cf: float = 2.0,       # C_HI / C_LO for HI tasks
    xf: float = 1.0,       # C_HI / C_LO for LO tasks
    beta: float = 0.3,
    seed: int = None,
    verbose: bool = True,
):
    """
    Run the full pipeline and return (processors, performance_dict).
    """
    if seed is not None:
        random.seed(seed)

    set_beta(beta)

    # ------------------------------------------------------------------
    # 1. Generate task set
    # ------------------------------------------------------------------
    tasks = generate_taskset(
        total_processor=num_processors,
        total_task=total_tasks,
        targetU=target_util,
        cp=cp,
        cf=cf,
        xf=xf,
    )

    lo_tasks_all = [t for t in tasks if t.criticality == "LO"]
    hi_tasks_all = [t for t in tasks if t.criticality == "HI"]

    if verbose:
        print(f"{'='*70}")
        print(f"Task set: {len(tasks)} tasks "
              f"({len(hi_tasks_all)} HI, {len(lo_tasks_all)} LO), "
              f"target U={target_util}, beta={beta}")
        print(f"{'='*70}")

    # ------------------------------------------------------------------
    # 2. Partition + classify
    # ------------------------------------------------------------------
    processors = partition_tasks(tasks, num_processors)
    if processors is None:
        if verbose:
            print("FAILED: partitioning infeasible.")
        return None

    # Snapshot drop_list after classification (for mode-switch performance)
    drops_for_mc: List[List[Task]] = [list(p.drop_list) for p in processors]

    if verbose:
        print()
        print("--- After Partition & Classification ---")
        for p in processors:
            lo = [t for t in p.tasks if t.criticality == "LO"]
            hi = [t for t in p.tasks if t.criticality == "HI"]
            print(f"  Core {p.id}: {len(hi)} HI, {len(lo)} LO  "
                  f"U_LO={p.utilization_lo:.3f}  U_HI={p.utilization_hi:.3f}  "
                  f"dropped={[t.id for t in p.drop_list]}")

    # ------------------------------------------------------------------
    # 3. LO-mode augmentation
    # ------------------------------------------------------------------
    for p in processors:
        lo_mode_augment(p.tasks, drop_list=p.drop_list, beta=beta)

    if verbose:
        print()
        print("--- After LO-Mode Augmentation ---")
        _print_x_table(processors, mode='L')

    # ------------------------------------------------------------------
    # 4. Mode-switch degradation
    # ------------------------------------------------------------------
    for p in processors:
        # mode_switch_degrade only operates on retained tasks
        mode_switch_degrade(p.tasks, drop_list=p.drop_list, beta=beta)

    if verbose:
        print()
        print("--- After Mode-Switch Degradation ---")
        _print_x_table(processors, mode='S')
        perfs = _compute_perfs(processors, lo_tasks_all, beta,
                               drops_for_mode_s=drops_for_mc)
        print(f"  Perf^L={perfs['L']:.4f}  Perf^S={perfs['S']:.4f}")

    # ------------------------------------------------------------------
    # 5. Stable-HI recovery
    # ------------------------------------------------------------------
    all_recovered: List[List[Task]] = []
    for p in processors:
        # Snapshot drops before recovery for reporting
        drops_before = set(t.id for t in p.drop_list)
        rec = stable_hi_recovery(p.tasks, p.drop_list, beta=beta)
        all_recovered.append(rec)

    if verbose:
        print()
        print("--- After Stable-HI Recovery & Post-Augment ---")
        _print_x_table(processors, mode='H')
        for p in processors:
            rec_ids = [t.id for t in all_recovered[p.id]]
            print(f"  Core {p.id}: recovered={rec_ids}, "
                  f"still suspended={[t.id for t in p.drop_list]}")

    # ------------------------------------------------------------------
    # 6. Performance summary
    # ------------------------------------------------------------------
    perfs = _compute_perfs(processors, lo_tasks_all, beta,
                           drops_for_mode_s=drops_for_mc)

    i_max = global_max_importance(lo_tasks_all, beta)

    if verbose:
        print()
        print(f"{'='*70}")
        print(f"Performance Summary")
        print(f"{'='*70}")
        print(f"  I^max = {i_max:.2f}")
        print(f"  Perf^L = {perfs['L']:.4f}   (LO mode)")
        print(f"  Perf^S = {perfs['S']:.4f}   (mode switch)")
        print(f"  Perf^H = {perfs['H']:.4f}   (stable HI)")
        print()
        # Per-core breakdown
        print("Per-core importance contribution:")
        header = f"  {'Core':<6} {'I^L':>10} {'I^S':>10} {'I^H':>10}"
        print(header)
        print(f"  {'-'*len(header)}")
        for p in processors:
            i_l = core_importance(p.tasks, p.drop_list, beta, 'L')  # no drops in L
            i_s = core_importance(p.tasks, drops_for_mc[p.id], beta, 'S')
            i_h = core_importance(p.tasks, p.drop_list, beta, 'H')
            print(f"  {p.id:<6} {i_l:>10.2f} {i_s:>10.2f} {i_h:>10.2f}")
        total_l = sum(core_importance(p.tasks, [], beta, 'L') for p in processors)
        total_s = sum(core_importance(p.tasks, drops_for_mc[p.id], beta, 'S')
                      for p in processors)
        total_h = sum(core_importance(p.tasks, p.drop_list, beta, 'H')
                      for p in processors)
        print(f"  {'Total':<6} {total_l:>10.2f} {total_s:>10.2f} {total_h:>10.2f}")
        print()

    return processors, perfs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_perfs(
    processors: List[Processor],
    all_lo_tasks: List[Task],
    beta: float,
    drops_for_mode_s: List[List[Task]] = None,
) -> dict:
    """Compute Perf^L, Perf^S, Perf^H."""
    i_max = global_max_importance(all_lo_tasks, beta)
    if i_max == 0:
        return {'L': 0.0, 'S': 0.0, 'H': 0.0}

    # Perf^L: all LO active (drop_list = []) → use empty drops
    total_l = sum(core_importance(p.tasks, [], beta, 'L') for p in processors)

    # Perf^S: use snapshotted drops from after classification
    total_s = sum(
        core_importance(p.tasks,
                        drops_for_mode_s[p.id] if drops_for_mode_s else p.drop_list,
                        beta, 'S')
        for p in processors
    )

    # Perf^H: use current drop_list (post-recovery)
    total_h = sum(core_importance(p.tasks, p.drop_list, beta, 'H')
                  for p in processors)

    return {
        'L': total_l / i_max,
        'S': total_s / i_max,
        'H': total_h / i_max,
    }


def _print_x_table(processors: List[Processor], mode: str):
    """Print x values for LO tasks per core."""
    x_attr = {'L': 'x_l', 'S': 'x_s', 'H': 'x_h'}[mode]
    for p in processors:
        lo = [t for t in p.tasks if t.criticality == "LO"]
        if not lo:
            continue
        entries = []
        for t in lo:
            x_val = getattr(t.mk, x_attr)
            dropped = "D" if t in p.drop_list else " "
            entries.append(f"T{t.id}(x={x_val}{dropped})")
        print(f"  Core {p.id}: {', '.join(entries)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline(
        num_processors=4,
        total_tasks=20,
        target_util=0.65,
        cp=0.5,
        cf=2.0,
        xf=1.0,
        beta=0.3,
        seed=41,
        verbose=True,
    )
