r"""
Beta sensitivity vs utilisation.

For each utilisation point, generates N random task sets.  Each feasible
task set is evaluated with all beta values {0.1, 0.3, 0.5, 0.7, 0.9} on
the SAME task set, isolating beta's effect.

Outputs CSV and line chart in experiments/beta_sensitivity/data/.
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

BETA_VALUES = [0.1, 0.3, 0.5, 0.7, 0.9]

UTIL_START = 0.40
UTIL_END   = 0.90
UTIL_STEP  = 0.05
N_RUNS     = 1000
N_FEASIBLE = 500          # target feasible sets per point
MAX_CONSECUTIVE_FAILS = 100  # stop if consecutive fails exceed this
NUM_THREADS = 5

OUTPUT_DIR = "experiments/beta_sensitivity/data"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "vary_beta.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "vary_beta.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deepcopy(processors: List[Processor]) -> List[Processor]:
    return copy.deepcopy(processors)


def _perf_h(processors: List[Processor], all_lo: List[Task], beta: float) -> float:
    i_max = global_max_importance(all_lo, beta)
    if i_max == 0:
        return 0.0
    total = sum(core_importance(p.tasks, p.drop_list, beta, 'H') for p in processors)
    return total / i_max


# ---------------------------------------------------------------------------
# Evaluate one task set under all beta values
# ---------------------------------------------------------------------------

def _eval_betas(processors: List[Processor],
                lo_all: List[Task],
                ) -> Dict[float, float]:
    """Run full pipeline for each beta, return {beta: Perf^H}."""
    result = {}
    for beta in BETA_VALUES:
        p = _deepcopy(processors)
        for proc in p:
            lo_mode_augment(proc.tasks, drop_list=proc.drop_list, beta=beta)
            mode_switch_degrade(proc.tasks, drop_list=proc.drop_list, beta=beta)
            stable_hi_recovery(proc.tasks, proc.drop_list, beta=beta)
        result[beta] = _perf_h(p, lo_all, beta)
    return result


# ---------------------------------------------------------------------------
# Single utilisation point
# ---------------------------------------------------------------------------

def run_util_point(target_util: float,
                   n_runs: int,
                   base_seed: int = 0) -> Dict[str, List[float]]:
    """
    Returns dict: {str(beta): [perf_values_across_runs], "feasible": count}
    """
    accum: Dict[float, float] = {b: 0.0 for b in BETA_VALUES}
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

        perfs = _eval_betas(processors, lo_all)
        for b in BETA_VALUES:
            accum[b] += perfs[b]

    cols = [str(b) for b in BETA_VALUES]
    if feasible == 0:
        result = {c: float('nan') for c in cols}
    else:
        result = {c: accum[float(c)] / feasible for c in cols}
    result["total_attempts"] = total_attempts
    result["feasible"] = feasible
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    util_values = []
    u = UTIL_START
    while u <= UTIL_END + 1e-9:
        util_values.append(u)
        u = round(u + UTIL_STEP, 10)

    beta_cols = [str(b) for b in BETA_VALUES]
    results: Dict[str, List[float]] = {c: [0.0] * len(util_values) for c in beta_cols}
    total_attempts = [0] * len(util_values)
    total_feasible = [0] * len(util_values)

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_idx = {}
        for i, u_val in enumerate(util_values):
            fut = executor.submit(run_util_point, u_val, N_RUNS, base_seed=0)
            future_to_idx[fut] = (i, u_val)

        for fut in as_completed(future_to_idx):
            i, u_val = future_to_idx[fut]
            perfs = fut.result()
            for c in beta_cols:
                results[c][i] = perfs[c]
            total_attempts[i] = perfs.get("total_attempts", 0)
            total_feasible[i] = perfs.get("feasible", 0)
            elapsed = time.time() - t0
            beta_strs = "  ".join(f"beta={b}:{perfs[b]:.4f}" for b in beta_cols)
            print(f"[{elapsed:7.1f}s]  U={u_val:.2f}  "
                  f"feas={total_feasible[i]}/{total_attempts[i]}  {beta_strs}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Utilization", "TotalAttempts", "Feasible"] + beta_cols)
        for i, u_val in enumerate(util_values):
            row = [f"{u_val:.2f}", str(total_attempts[i]), str(total_feasible[i])]
            for c in beta_cols:
                row.append(f"{results[c][i]:.6f}")
            writer.writerow(row)
    print(f"Saved: {OUTPUT_CSV}")

    # ---- Plot ----
    _plot(util_values, results, OUTPUT_PLOT)
    print(f"Saved: {OUTPUT_PLOT}")


def _plot(util_values, results, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    # color gradient from cool (low beta) to warm (high beta)
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#E91E63", "#9C27B0"]
    beta_strs = [str(b) for b in BETA_VALUES]

    for i, b_str in enumerate(beta_strs):
        plt.plot(util_values, results[b_str],
                 color=colors[i], marker="o", linewidth=1.5, markersize=4,
                 label=f"$\\beta = {b_str}$")

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
        N_FEASIBLE = 1
    main()
