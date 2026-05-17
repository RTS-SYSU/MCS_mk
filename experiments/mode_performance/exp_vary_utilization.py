r"""
Mode-wise performance vs utilisation.

Varies target utilisation from 0.40 to 0.90 (step 0.05).
For each point, generates N random task sets; tracks:
  - Feasibility rate (fraction of successfully partitioned sets)
  - Perf^L, Perf^S, Perf^H for Proposed and Static-(m,k)

Outputs CSV and line chart in experiments/mode_performance/data/.
"""

import copy
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

from core.importance import set_beta
from core.processor import Processor
from core.task import Task
from scheduling.task_partitioning import partition_tasks
from scheduling.augmentation import lo_mode_augment, mode_switch_degrade
from scheduling.recovery import stable_hi_recovery
from experiments.performance import global_max_importance, core_importance
from utils.generate_taskset import generate_taskset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_PROCESSORS = 4
TASKS_PER_CORE = 20
TOTAL_TASKS = NUM_PROCESSORS * TASKS_PER_CORE
CP = 0.5
CF = 2.0
XF = 1.0
BETA = 0.5

UTIL_START = 0.40
UTIL_END   = 0.90
UTIL_STEP  = 0.05
N_RUNS     = 1000
N_FEASIBLE = 500          # target feasible sets per point
MAX_CONSECUTIVE_FAILS = 100  # stop if consecutive fails exceed this
NUM_THREADS = 5

OUTPUT_DIR = "experiments/mode_performance/data"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "vary_utilization.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "vary_utilization.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deepcopy_procs(processors: List[Processor]) -> List[Processor]:
    return copy.deepcopy(processors)


def _perf_mode(processors: List[Processor],
               all_lo: List[Task],
               beta: float,
               mode: str,
               drops_override: List[List[Task]] = None) -> float:
    """Perf^chi for all cores."""
    i_max = global_max_importance(all_lo, beta)
    if i_max == 0:
        return 0.0
    total = 0.0
    for p in processors:
        drops = drops_override[p.id] if drops_override else p.drop_list
        total += core_importance(p.tasks, drops, beta, mode)
    return total / i_max


# ---------------------------------------------------------------------------
# Single utilisation point
# ---------------------------------------------------------------------------

def run_util_point(target_util: float,
                   n_runs: int,
                   beta: float,
                   base_seed: int = 0) -> Dict[str, float]:
    """
    Returns dict with keys:
      feasible, Static_L, Static_S, Static_H,
      Proposed_L, Proposed_S, Proposed_H
    """
    accum = {"Static_L": 0.0, "Static_S": 0.0, "Static_H": 0.0,
             "Proposed_L": 0.0, "Proposed_S": 0.0, "Proposed_H": 0.0}
    feasible = 0
    consec_fails = 0
    total_attempts = 0

    while feasible < N_FEASIBLE and consec_fails < MAX_CONSECUTIVE_FAILS:
        total_attempts += 1
        random.seed(base_seed + int(target_util * 10000) + total_attempts)

        tasks = generate_taskset(
            total_processor=NUM_PROCESSORS, total_task=TOTAL_TASKS,
            targetU=target_util, cp=CP, cf=CF, xf=XF,
        )
        lo_all = [t for t in tasks if t.criticality == "LO"]

        processors = partition_tasks(tasks, NUM_PROCESSORS)
        if processors is None:
            consec_fails += 1
            continue
        consec_fails = 0
        feasible += 1

        drops_mc = [list(p.drop_list) for p in processors]

        p_static = _deepcopy_procs(processors)
        for p in p_static:
            for t in p.tasks:
                if t.criticality == "LO":
                    t.mk.set_x(0, 'L')
                    t.mk.set_x(0, 'S')
                    t.mk.set_x(0, 'H')
        accum["Static_L"] += _perf_mode(p_static, lo_all, beta, 'L')
        accum["Static_S"] += _perf_mode(p_static, lo_all, beta, 'S', drops_mc)
        accum["Static_H"] += _perf_mode(p_static, lo_all, beta, 'H', drops_mc)

        base = _deepcopy_procs(processors)
        for p in base:
            lo_mode_augment(p.tasks, drop_list=p.drop_list, beta=beta)
        accum["Proposed_L"] += _perf_mode(base, lo_all, beta, 'L')
        for p in base:
            mode_switch_degrade(p.tasks, drop_list=p.drop_list, beta=beta)
        accum["Proposed_S"] += _perf_mode(base, lo_all, beta, 'S', drops_mc)
        for p in base:
            stable_hi_recovery(p.tasks, p.drop_list, beta=beta)
        accum["Proposed_H"] += _perf_mode(base, lo_all, beta, 'H')

    if feasible == 0:
        result = {"Static_L": float('nan'), "Static_S": float('nan'), "Static_H": float('nan'),
                  "Proposed_L": float('nan'), "Proposed_S": float('nan'), "Proposed_H": float('nan')}
    else:
        result = {"Static_L":   accum["Static_L"]   / feasible,
                  "Static_S":   accum["Static_S"]   / feasible,
                  "Static_H":   accum["Static_H"]   / feasible,
                  "Proposed_L": accum["Proposed_L"] / feasible,
                  "Proposed_S": accum["Proposed_S"] / feasible,
                  "Proposed_H": accum["Proposed_H"] / feasible}
    result["total_attempts"] = total_attempts
    result["feasible"] = feasible
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_beta(BETA)

    util_values = []
    u = UTIL_START
    while u <= UTIL_END + 1e-9:
        util_values.append(u)
        u = round(u + UTIL_STEP, 10)

    perf_cols = ["Static_L", "Static_S", "Static_H",
                 "Proposed_L", "Proposed_S", "Proposed_H"]
    results: Dict[str, List[float]] = {c: [0.0] * len(util_values) for c in perf_cols}
    total_attempts = [0] * len(util_values)
    total_feasible = [0] * len(util_values)

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_idx = {}
        for i, u_val in enumerate(util_values):
            fut = executor.submit(run_util_point, u_val, N_RUNS, BETA, base_seed=0)
            future_to_idx[fut] = (i, u_val)

        for fut in as_completed(future_to_idx):
            i, u_val = future_to_idx[fut]
            perfs = fut.result()
            for c in perf_cols:
                results[c][i] = perfs[c]
            total_attempts[i] = perfs.get("total_attempts", 0)
            total_feasible[i] = perfs.get("feasible", 0)
            elapsed = time.time() - t0
            print(f"[{elapsed:7.1f}s]  U={u_val:.2f}  "
                  f"feas={total_feasible[i]}/{total_attempts[i]}  "
                  f"Prop(L={perfs['Proposed_L']:.4f} S={perfs['Proposed_S']:.4f} H={perfs['Proposed_H']:.4f})  "
                  f"Stat(L={perfs['Static_L']:.4f} S={perfs['Static_S']:.4f} H={perfs['Static_H']:.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Utilization", "TotalAttempts", "Feasible"] + perf_cols)
        for i, u_val in enumerate(util_values):
            row = [f"{u_val:.2f}", str(total_attempts[i]), str(total_feasible[i])]
            for c in perf_cols:
                row.append(f"{results[c][i]:.6f}")
            writer.writerow(row)
    print(f"Saved: {OUTPUT_CSV}")

    # ---- Plot ----
    _plot(util_values, results, OUTPUT_PLOT)
    print(f"Saved: {OUTPUT_PLOT}")


def _plot(util_values, results, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    modes = [
        ("Proposed_L", "Proposed L",  "#2196F3", "-",  "o"),
        ("Proposed_S", "Proposed S",  "#FF9800", "-",  "s"),
        ("Proposed_H", "Proposed H",  "#E91E63", "-",  "D"),
        ("Static_L",   "Static L",    "#90CAF9", "--", "o"),
        ("Static_S",   "Static S",    "#FFCC80", "--", "s"),
        ("Static_H",   "Static H",    "#F48FB1", "--", "D"),
    ]
    for key, label, color, ls, mk in modes:
        plt.plot(util_values, results[key],
                 color=color, linestyle=ls, marker=mk,
                 linewidth=1.5, markersize=4, label=label)

    plt.xlabel("Target utilisation $U$")
    plt.ylabel("Normalised performance")
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        print("=== Quick test mode (1 run per point) ===")
        N_FEASIBLE = 1
    main()
