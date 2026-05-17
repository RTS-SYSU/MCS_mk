r"""
CP-variation experiment (fraction of HI-criticality tasks).

Varies CP from 0.1 to 0.9 (step 0.1).  More HI tasks means more tasks
that may expand execution in mode switch.

For each CP, generates random task sets until N_FEASIBLE feasible sets
are found or MAX_CONSECUTIVE_FAILS consecutive partitions fail.

Compares four methods on Perf^H (normalized stable-HI performance):
  - Static-(m,k)
  - Augmented-Only
  - MaxCount Recovery
  - Proposed

Outputs a CSV table and a line chart in experiments/vary_cp/data/.
"""

import copy
import csv
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List

from core.importance import set_beta
from core.processor import Processor
from core.task import Task
from scheduling.task_partitioning import partition_tasks
from scheduling.augmentation import lo_mode_augment, mode_switch_degrade
from scheduling.recovery import (
    stable_hi_recovery,
    _hi_mode_augment,
)
from experiments.comparison.maxcount_recovery import _maxcount_recover_core
from experiments.performance import (
    global_max_importance,
    core_importance,
)
from utils.generate_taskset import generate_taskset


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

NUM_PROCESSORS = 4
TASKS_PER_CORE = 20
TOTAL_TASKS = NUM_PROCESSORS * TASKS_PER_CORE  # 80
CF = 2.0        # C_HI / C_LO for HI tasks
XF = 1.0        # C_HI / C_LO for LO tasks
BETA = 0.5
TARGET_U = 0.65  # fixed utilisation

CP_START = 0.1
CP_END = 0.9
CP_STEP = 0.1
N_RUNS = 20000
N_FEASIBLE = 500          # target feasible sets per point
MAX_CONSECUTIVE_FAILS = 1000  # stop if consecutive fails exceed this
NUM_THREADS = 50

OUTPUT_DIR = "experiments/vary_cp/data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "vary_cp.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "vary_cp.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perf_h(processors: List[Processor],
            all_lo_tasks: List[Task],
            beta: float) -> float:
    i_max = global_max_importance(all_lo_tasks, beta)
    if i_max == 0:
        return 0.0
    total = sum(core_importance(p.tasks, p.drop_list, beta, 'H')
                for p in processors)
    return total / i_max


def _deepcopy_procs(processors: List[Processor]) -> List[Processor]:
    return copy.deepcopy(processors)


# ---------------------------------------------------------------------------
# Single CP point
# ---------------------------------------------------------------------------

def run_cp_point(cp: float,
                 n_runs: int,
                 beta: float,
                 base_seed: int = 0) -> Dict[str, float]:
    methods = ["Static", "AugOnly", "MaxCount", "Proposed"]
    accum: Dict[str, float] = {m: 0.0 for m in methods}
    success = 0
    consec_fails = 0
    total_attempts = 0

    while success < N_FEASIBLE and consec_fails < MAX_CONSECUTIVE_FAILS:
        total_attempts += 1

        random.seed(base_seed + int(cp * 10000) + total_attempts)

        tasks = generate_taskset(
            total_processor=NUM_PROCESSORS,
            total_task=TOTAL_TASKS,
            targetU=TARGET_U,
            cp=cp,
            cf=CF,
            xf=XF,
        )
        lo_all = [t for t in tasks if t.criticality == "LO"]

        processors = partition_tasks(tasks, NUM_PROCESSORS)
        if processors is None:
            consec_fails += 1
            continue

        consec_fails = 0
        success += 1

        base = _deepcopy_procs(processors)
        for p in base:
            lo_mode_augment(p.tasks, drop_list=p.drop_list, beta=beta)
        for p in base:
            mode_switch_degrade(p.tasks, drop_list=p.drop_list, beta=beta)

        p_static = _deepcopy_procs(processors)
        for p in p_static:
            for t in p.tasks:
                if t.criticality == "LO":
                    t.mk.set_x(0, 'L')
                    t.mk.set_x(0, 'S')
                    t.mk.set_x(0, 'H')
        accum["Static"] += _perf_h(p_static, lo_all, beta)

        p_aug = _deepcopy_procs(base)
        for p in p_aug:
            _hi_mode_augment(p.tasks, p.drop_list, beta)
        accum["AugOnly"] += _perf_h(p_aug, lo_all, beta)

        p_mc = _deepcopy_procs(base)
        for p in p_mc:
            _maxcount_recover_core(p.tasks, p.drop_list, beta)
        accum["MaxCount"] += _perf_h(p_mc, lo_all, beta)

        p_prop = _deepcopy_procs(base)
        for p in p_prop:
            stable_hi_recovery(p.tasks, p.drop_list, beta=beta)
        accum["Proposed"] += _perf_h(p_prop, lo_all, beta)

    if success == 0:
        result = {m: float('nan') for m in methods}
    else:
        result = {m: accum[m] / success for m in methods}
    result["total_attempts"] = total_attempts
    result["feasible"] = success
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_beta(BETA)

    cp_values = []
    cp = CP_START
    while cp <= CP_END + 1e-9:
        cp_values.append(cp)
        cp = round(cp + CP_STEP, 10)

    methods = ["Static", "AugOnly", "MaxCount", "Proposed"]
    results: Dict[str, List[float]] = {m: [0.0] * len(cp_values) for m in methods}
    total_attempts = [0] * len(cp_values)
    total_feasible = [0] * len(cp_values)

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_idx = {}
        for i, cp_val in enumerate(cp_values):
            fut = executor.submit(run_cp_point, cp_val, N_RUNS, BETA, base_seed=1)
            future_to_idx[fut] = (i, cp_val)

        for fut in as_completed(future_to_idx):
            i, cp_val = future_to_idx[fut]
            perfs = fut.result()
            for m in methods:
                results[m][i] = perfs[m]
            total_attempts[i] = perfs.get("total_attempts", 0)
            total_feasible[i] = perfs.get("feasible", 0)
            elapsed = time.time() - t0
            print(f"[{elapsed:6.1f}s]  CP={cp_val:.1f}  "
                  f"feas={total_feasible[i]}/{total_attempts[i]}  "
                  f"Static={perfs['Static']:.4f}  "
                  f"AugOnly={perfs['AugOnly']:.4f}  "
                  f"MaxCount={perfs['MaxCount']:.4f}  "
                  f"Proposed={perfs['Proposed']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["CP", "TotalAttempts", "Feasible"] + methods)
        for i, cp_val in enumerate(cp_values):
            row = [f"{cp_val:.1f}", str(total_attempts[i]), str(total_feasible[i])]
            for m in methods:
                row.append(f"{results[m][i]:.6f}")
            writer.writerow(row)
    print(f"Saved: {OUTPUT_CSV}")

    _plot(cp_values, results, OUTPUT_PLOT)
    print(f"Saved: {OUTPUT_PLOT}")


def _plot(cp_values, results, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    markers = {"Static": "s", "AugOnly": "o", "MaxCount": "^", "Proposed": "D"}
    colors = {"Static": "gray", "AugOnly": "#2196F3", "MaxCount": "#FF9800",
              "Proposed": "#E91E63"}

    for method, perfs in results.items():
        plt.plot(cp_values, perfs,
                 marker=markers.get(method, "x"),
                 color=colors.get(method, "black"),
                 linewidth=1.5, markersize=5, label=method)

    plt.xlabel("CP  (fraction of HI-criticality tasks)")
    plt.ylabel("Normalised performance $\\mathrm{Perf}^{\\mathrm{H}}$")
    plt.ylim(-0.02, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
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
