r"""
Utilization-variation experiment.

Varies target utilization from 0.40 to 0.90 (step 0.05).
For each utilization, generates N random task sets; only successfully
partitioned + classified sets are counted.

Compares four methods on Perf^H (normalized stable-HI performance):
  - Static-(m,k)
  - Augmented-Only
  - MaxCount Recovery
  - Proposed

Outputs a CSV table and a line chart in the data/ folder.
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
CP = 0.5        # fraction of HI tasks
CF = 2.0        # C_HI / C_LO for HI tasks
XF = 1.0        # C_HI / C_LO for LO tasks
BETA = 0.5

UTIL_START = 0.40
UTIL_END = 0.90
UTIL_STEP = 0.05
N_RUNS = 10000       # random task sets per utilisation point
N_FEASIBLE = 500       # target number of feasible task sets
MAX_CONSECUTIVE_FAILS = 500  # stop if this many consecutive partitions fail
NUM_THREADS = 150     # process pool size

OUTPUT_DIR = "experiments/vary_utilization/data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "vary_utilization.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "vary_utilization.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _perf_h(processors: List[Processor],
            all_lo_tasks: List[Task],
            beta: float) -> float:
    """Compute Perf^H for a set of processors."""
    i_max = global_max_importance(all_lo_tasks, beta)
    if i_max == 0:
        return 0.0
    total = sum(core_importance(p.tasks, p.drop_list, beta, 'H')
                for p in processors)
    return total / i_max


def _deepcopy_procs(processors: List[Processor]) -> List[Processor]:
    return copy.deepcopy(processors)


# ---------------------------------------------------------------------------
# Single utilisation point (thread-safe, uses local random seed)
# ---------------------------------------------------------------------------

def run_util_point(target_util: float,
                   n_runs: int,
                   beta: float,
                   base_seed: int = 0) -> Dict[str, float]:
    """
    Run N_RUNS random task sets at a given target utilisation.

    Returns dict {method_name: avg_Perf_H} across successful task sets.
    """
    rng = random.Random(base_seed + int(target_util * 10000))

    methods = ["Static", "AugOnly", "MaxCount", "Proposed"]
    accum: Dict[str, float] = {m: 0.0 for m in methods}
    success = 0
    consec_fails = 0
    total_attempts = 0

    while success < N_FEASIBLE and consec_fails < MAX_CONSECUTIVE_FAILS:
        rng.seed(base_seed + int(target_util * 10000) + total_attempts)
        total_attempts += 1

        # 1. Generate
        tasks = generate_taskset(
            total_processor=NUM_PROCESSORS,
            total_task=TOTAL_TASKS,
            targetU=target_util,
            cp=CP,
            cf=CF,
            xf=XF,
        )
        lo_all = [t for t in tasks if t.criticality == "LO"]

        # 2. Partition
        processors = partition_tasks(tasks, NUM_PROCESSORS)
        if processors is None:
            consec_fails += 1
            continue

        consec_fails = 0
        success += 1

        # 3. Shared base: LO augment + MC degrade
        base = _deepcopy_procs(processors)
        for p in base:
            lo_mode_augment(p.tasks, drop_list=p.drop_list, beta=beta)
        for p in base:
            mode_switch_degrade(p.tasks, drop_list=p.drop_list, beta=beta)

        # 4a. Static: from partitioned state, x = 0
        p_static = _deepcopy_procs(processors)
        for p in p_static:
            for t in p.tasks:
                if t.criticality == "LO":
                    t.mk.set_x(0, 'L')
                    t.mk.set_x(0, 'S')
                    t.mk.set_x(0, 'H')
        accum["Static"] += _perf_h(p_static, lo_all, beta)

        # 4b. AugOnly: from post-MC, post-augment only, no recovery
        p_aug = _deepcopy_procs(base)
        for p in p_aug:
            _hi_mode_augment(p.tasks, p.drop_list, beta)
        accum["AugOnly"] += _perf_h(p_aug, lo_all, beta)

        # 4c. MaxCount: from post-MC, maxcount recovery, no post-augment
        p_mc = _deepcopy_procs(base)
        for p in p_mc:
            _maxcount_recover_core(p.tasks, p.drop_list, beta)
        accum["MaxCount"] += _perf_h(p_mc, lo_all, beta)

        # 4d. Proposed: from post-MC, full recovery + post-augment
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

    util_values = []
    u = UTIL_START
    while u <= UTIL_END + 1e-9:
        util_values.append(u)
        u = round(u + UTIL_STEP, 10)

    methods = ["Static", "AugOnly", "MaxCount", "Proposed"]
    results: Dict[str, List[float]] = {m: [0.0] * len(util_values) for m in methods}
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
            for m in methods:
                results[m][i] = perfs[m]
            total_attempts[i] = perfs.get("total_attempts", 0)
            total_feasible[i] = perfs.get("feasible", 0)
            elapsed = time.time() - t0
            print(f"[{elapsed:6.1f}s]  U={u_val:.2f}  "
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
        writer.writerow(["Utilization", "TotalAttempts", "Feasible"] + methods)
        for i, u_val in enumerate(util_values):
            row = [f"{u_val:.2f}", str(total_attempts[i]), str(total_feasible[i])]
            for m in methods:
                row.append(f"{results[m][i]:.6f}")
            writer.writerow(row)
    print(f"Saved: {OUTPUT_CSV}")

    # ---- Plot ----
    _plot(util_values, results, OUTPUT_PLOT)
    print(f"Saved: {OUTPUT_PLOT}")


def _plot(util_values, results, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    markers = {"Static": "s", "AugOnly": "o", "MaxCount": "^", "Proposed": "D"}
    colors = {"Static": "gray", "AugOnly": "#2196F3", "MaxCount": "#FF9800",
              "Proposed": "#E91E63"}

    for method, perfs in results.items():
        plt.plot(util_values, perfs,
                 marker=markers.get(method, "x"),
                 color=colors.get(method, "black"),
                 linewidth=1.5, markersize=5, label=method)

    plt.xlabel("Target utilisation $U$")
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
        N_RUNS = 1
    print(f"=== (utilization: {N_RUNS} runs per point) ===")
    main()
