r"""
Static-AMC-WH vs Proposed: per-mode performance vs utilisation.

Static-AMC-WH: x_l = k-m (full), x_s = x_h = 0.  Drop in MC, no recovery.
Proposed:     full pipeline (LO augment → MC degrade → HI recovery).

If a mode is not schedulable, Perf^mode = 0 for that method.

Outputs CSV and line chart in experiments/amc_wh_comparison/data/.
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
from experiments.comparison.static_amc_wh import run as run_static_amc
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
NUM_THREADS = 5

OUTPUT_DIR = "experiments/amc_wh_comparison/data"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "vary_utilization.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "vary_utilization.png")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deepcopy(processors: List[Processor]) -> List[Processor]:
    return copy.deepcopy(processors)


def _perf_mode(processors, all_lo, beta, mode, drops_override=None):
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
    Returns: feasible, + 6 perf columns:
      Static_L, Static_S, Static_H, Proposed_L, Proposed_S, Proposed_H
    """
    accum = {"Static_L": 0.0, "Static_S": 0.0, "Static_H": 0.0,
             "Proposed_L": 0.0, "Proposed_S": 0.0, "Proposed_H": 0.0}
    feasible = 0

    for run_idx in range(n_runs):
        random.seed(base_seed + int(target_util * 10000) + run_idx)
        tasks = generate_taskset(
            total_processor=NUM_PROCESSORS, total_task=TOTAL_TASKS,
            targetU=target_util, cp=CP, cf=CF, xf=XF,
        )
        lo_all = [t for t in tasks if t.criticality == "LO"]

        processors = partition_tasks(tasks, NUM_PROCESSORS)
        if processors is None:
            continue
        feasible += 1

        # ---- Static-AMC-WH ----
        p_static = _deepcopy(processors)
        perfs = run_static_amc(p_static, lo_all, beta)
        accum["Static_L"] += perfs["L"]
        accum["Static_S"] += perfs["S"]
        accum["Static_H"] += perfs["H"]

        # ---- Proposed ----
        p_prop = _deepcopy(processors)
        for proc in p_prop:
            lo_mode_augment(proc.tasks, drop_list=proc.drop_list, beta=beta)

        accum["Proposed_L"] += _perf_mode(p_prop, lo_all, beta, 'L')

        drops_mc = [list(proc.drop_list) for proc in p_prop]
        for proc in p_prop:
            mode_switch_degrade(proc.tasks, drop_list=proc.drop_list, beta=beta)

        accum["Proposed_S"] += _perf_mode(p_prop, lo_all, beta, 'S', drops_mc)

        for proc in p_prop:
            stable_hi_recovery(proc.tasks, proc.drop_list, beta=beta)

        accum["Proposed_H"] += _perf_mode(p_prop, lo_all, beta, 'H')

    cols = ["Static_L", "Static_S", "Static_H",
            "Proposed_L", "Proposed_S", "Proposed_H"]
    if feasible == 0:
        return {"feasible": 0, **{c: float('nan') for c in cols}}

    return {"feasible": feasible, **{c: accum[c] / feasible for c in cols}}


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

    cols = ["feasible", "Static_L", "Static_S", "Static_H",
            "Proposed_L", "Proposed_S", "Proposed_H"]
    results: Dict[str, List[float]] = {c: [0.0] * len(util_values) for c in cols}

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_idx = {}
        for i, u_val in enumerate(util_values):
            fut = executor.submit(run_util_point, u_val, N_RUNS, BETA, base_seed=0)
            future_to_idx[fut] = (i, u_val)

        for fut in as_completed(future_to_idx):
            i, u_val = future_to_idx[fut]
            r = fut.result()
            for c in cols:
                results[c][i] = r[c]
            rate = r["feasible"] / N_RUNS * 100
            elapsed = time.time() - t0
            print(f"[{elapsed:7.1f}s]  U={u_val:.2f}  feas={rate:.1f}%  "
                  f"Stat(L={r['Static_L']:.4f} S={r['Static_S']:.4f} H={r['Static_H']:.4f})  "
                  f"Prop(L={r['Proposed_L']:.4f} S={r['Proposed_S']:.4f} H={r['Proposed_H']:.4f})")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # ---- Save CSV ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Utilization", "FeasibilityRate"] + cols[1:])
        for i, u_val in enumerate(util_values):
            row = [f"{u_val:.2f}", f"{results['feasible'][i] / N_RUNS:.6f}"]
            for c in cols[1:]:
                row.append(f"{results[c][i]:.6f}")
            writer.writerow(row)
    print(f"Saved: {OUTPUT_CSV}")

    # ---- Plot ----
    _plot(util_values, results, OUTPUT_PLOT)
    print(f"Saved: {OUTPUT_PLOT}")


def _plot(util_values, results, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, mode, label in [
        (axes[0], 'L', 'LO mode'),
        (axes[1], 'S', 'Mode switch'),
        (axes[2], 'H', 'Stable HI'),
    ]:
        key_p = f"Proposed_{mode}"
        key_s = f"Static_{mode}"
        ax.plot(util_values, results[key_p], color="#E91E63", marker="D",
                linewidth=1.5, markersize=4, label="Proposed")
        ax.plot(util_values, results[key_s], color="#607D8B", marker="s",
                linewidth=1.5, markersize=4, linestyle="--", label="Static-AMC-WH")
        ax.set_xlabel("Target utilisation $U$")
        ax.set_title(label)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Normalised performance")
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
    main()
