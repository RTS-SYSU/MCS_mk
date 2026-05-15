r"""
Suspended count & recovery success rate vs utilisation.

For each utilisation point, generates N random task sets and tracks:
  - Avg # of suspended LO tasks (after classification)
  - Avg # of recovered LO tasks (after stable-HI recovery)
  - Recovery success rate (recovered / suspended)
  - Perf^H for Proposed

Outputs CSV, LaTeX table, and line chart in experiments/suspend_recovery/data/.
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
BETA = 0.5

UTIL_START = 0.40
UTIL_END   = 0.90
UTIL_STEP  = 0.05
N_RUNS     = 1000
NUM_THREADS = 5

OUTPUT_DIR = "experiments/suspend_recovery/data"
OUTPUT_CSV  = os.path.join(OUTPUT_DIR, "suspend_recovery.csv")
OUTPUT_TEX  = os.path.join(OUTPUT_DIR, "suspend_recovery.tex")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "suspend_recovery.png")


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
# Single utilisation point
# ---------------------------------------------------------------------------

def run_util_point(target_util: float,
                   n_runs: int,
                   base_seed: int = 0) -> Dict[str, float]:
    """
    Returns: feasible, suspended, recovered, rec_rate, Perf_H
    """
    accum = {"suspended": 0.0, "recovered": 0.0, "Perf_H": 0.0}
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

        # Count suspended (after classification, before recovery)
        n_suspended = sum(len(p.drop_list) for p in processors)

        # LO augment + MC degrade
        for p in processors:
            lo_mode_augment(p.tasks, drop_list=p.drop_list, beta=BETA)
            mode_switch_degrade(p.tasks, drop_list=p.drop_list, beta=BETA)

        # HI recovery — returns list of recovered tasks
        n_recovered = 0
        for p in processors:
            rec = stable_hi_recovery(p.tasks, p.drop_list, beta=BETA)
            n_recovered += len(rec)

        accum["suspended"] += n_suspended
        accum["recovered"] += n_recovered
        accum["Perf_H"] += _perf_h(processors, lo_all, BETA)

    if feasible == 0:
        return {"feasible": 0,
                "suspended": float('nan'), "recovered": float('nan'),
                "rec_rate": float('nan'), "Perf_H": float('nan')}

    return {"feasible": feasible,
            "suspended":  accum["suspended"]  / feasible,
            "recovered":  accum["recovered"]  / feasible,
            "rec_rate":   accum["recovered"] / accum["suspended"] if accum["suspended"] > 0 else 1.0,
            "Perf_H":     accum["Perf_H"]     / feasible}


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

    cols = ["feasible", "suspended", "recovered", "rec_rate", "Perf_H"]
    results: Dict[str, List[float]] = {c: [0.0] * len(util_values) for c in cols}

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_idx = {}
        for i, u_val in enumerate(util_values):
            fut = executor.submit(run_util_point, u_val, N_RUNS, base_seed=0)
            future_to_idx[fut] = (i, u_val)

        for fut in as_completed(future_to_idx):
            i, u_val = future_to_idx[fut]
            r = fut.result()
            for c in cols:
                results[c][i] = r[c]
            rate = r["feasible"] / N_RUNS * 100
            elapsed = time.time() - t0
            print(f"[{elapsed:7.1f}s]  U={u_val:.2f}  feas={rate:.1f}%  "
                  f"susp={r['suspended']:.1f}  rec={r['recovered']:.1f}  "
                  f"rate={r['rec_rate']*100:.1f}%  Perf_H={r['Perf_H']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    # ---- Save CSV ----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Utilization", "FeasibilityRate", "Suspended",
                          "Recovered", "RecoveryRate", "Perf_H"])
        for i, u_val in enumerate(util_values):
            writer.writerow([
                f"{u_val:.2f}",
                f"{results['feasible'][i] / N_RUNS:.6f}",
                f"{results['suspended'][i]:.4f}",
                f"{results['recovered'][i]:.4f}",
                f"{results['rec_rate'][i]:.4f}",
                f"{results['Perf_H'][i]:.6f}",
            ])
    print(f"Saved: {OUTPUT_CSV}")

    # ---- Save LaTeX ----
    with open(OUTPUT_TEX, "w") as f:
        f.write(r"\begin{table}[htbp]" + "\n")
        f.write(r"  \centering" + "\n")
        f.write(r"  \caption{Suspended tasks, recovery count, recovery rate, "
                r"and $\mathrm{Perf}^{\mathrm{H}}$ across utilisation.}" + "\n")
        f.write(r"  \label{tab:suspend_recovery}" + "\n")
        f.write(r"  \begin{tabular}{cccccc}" + "\n")
        f.write(r"    \toprule" + "\n")
        f.write(r"    $U$ & Feas. & Susp. & Recov. & Rec.Rate & $\mathrm{Perf}^{\mathrm{H}}$ \\" + "\n")
        f.write(r"    \midrule" + "\n")
        for i, u_val in enumerate(util_values):
            fr = results["feasible"][i] / N_RUNS
            if fr == 0:
                continue
            f.write(f"    {u_val:.2f} & {fr:.2f} & "
                    f"{results['suspended'][i]:.1f} & "
                    f"{results['recovered'][i]:.1f} & "
                    f"{results['rec_rate'][i]*100:.0f}\\% & "
                    f"{results['Perf_H'][i]:.4f} \\\\\n")
        f.write(r"    \bottomrule" + "\n")
        f.write(r"  \end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n")
    print(f"Saved: {OUTPUT_TEX}")

    # ---- Plot ----
    _plot(util_values, results, OUTPUT_PLOT)
    print(f"Saved: {OUTPUT_PLOT}")


def _plot(util_values, results, path):
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: suspended / recovered count
    ax1.plot(util_values, results["suspended"], color="#E91E63",
             marker="o", linewidth=1.5, markersize=4, label="Suspended")
    ax1.plot(util_values, results["recovered"], color="#4CAF50",
             marker="s", linewidth=1.5, markersize=4, label="Recovered")
    ax1.set_xlabel("Target utilisation $U$")
    ax1.set_ylabel("Avg. number of tasks")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Right: recovery rate
    ax2.plot(util_values,
             [r * 100 for r in results["rec_rate"]],
             color="#2196F3", marker="D", linewidth=1.5, markersize=4)
    ax2.set_xlabel("Target utilisation $U$")
    ax2.set_ylabel("Recovery success rate (%)")
    ax2.set_ylim(-2, 105)
    ax2.grid(True, alpha=0.3)

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
