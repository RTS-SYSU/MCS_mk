r"""
(m,k)-constraint variation experiment.

Varies m from 1 to 9 with k=10 fixed.  Higher m/k means more mandatory
jobs (higher baseline importance, higher LO utilisation).

For each m, generates N random task sets; only successfully partitioned
+ classified sets are counted.

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
TARGET_U = 0.65

K_FIXED = 10
M_START = 1
M_END = 9
M_STEP = 1

N_RUNS = 10000
NUM_THREADS = 10

OUTPUT_DIR = "experiments/vary_mk/data"
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "vary_mk.csv")
OUTPUT_PLOT = os.path.join(OUTPUT_DIR, "vary_mk.png")


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
# Single m point
# ---------------------------------------------------------------------------

def run_m_point(m: int,
                n_runs: int,
                beta: float,
                base_seed: int = 0) -> Dict[str, float]:
    methods = ["Static", "AugOnly", "MaxCount", "Proposed"]
    accum: Dict[str, float] = {mtd: 0.0 for mtd in methods}
    success = 0

    for run_idx in range(n_runs):
        random.seed(base_seed + m * 10000 + run_idx)

        tasks = generate_taskset(
            total_processor=NUM_PROCESSORS,
            total_task=TOTAL_TASKS,
            targetU=TARGET_U,
            cp=CP,
            cf=CF,
            xf=XF,
            m=m,
            k=K_FIXED,
        )
        lo_all = [t for t in tasks if t.criticality == "LO"]

        processors = partition_tasks(tasks, NUM_PROCESSORS)
        if processors is None:
            continue

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

        success += 1

    if success == 0:
        return {mtd: float('nan') for mtd in methods}
    return {mtd: accum[mtd] / success for mtd in methods}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    set_beta(BETA)

    m_values = list(range(M_START, M_END + 1, M_STEP))

    results: Dict[str, List[float]] = {
        "Static": [0.0] * len(m_values),
        "AugOnly": [0.0] * len(m_values),
        "MaxCount": [0.0] * len(m_values),
        "Proposed": [0.0] * len(m_values),
    }

    t0 = time.time()

    with ProcessPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_idx = {}
        for i, m_val in enumerate(m_values):
            fut = executor.submit(run_m_point, m_val, N_RUNS, BETA, base_seed=2)
            future_to_idx[fut] = (i, m_val)

        for fut in as_completed(future_to_idx):
            i, m_val = future_to_idx[fut]
            perfs = fut.result()
            for mtd in results:
                results[mtd][i] = perfs[mtd]
            elapsed = time.time() - t0
            print(f"[{elapsed:6.1f}s]  m={m_val} (k={K_FIXED})  "
                  f"Static={perfs['Static']:.4f}  "
                  f"AugOnly={perfs['AugOnly']:.4f}  "
                  f"MaxCount={perfs['MaxCount']:.4f}  "
                  f"Proposed={perfs['Proposed']:.4f}")

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.1f}s")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["m"] + list(results.keys()))
        for i, m_val in enumerate(m_values):
            row = [str(m_val)]
            for mtd in results:
                row.append(f"{results[mtd][i]:.6f}")
            writer.writerow(row)
    print(f"Saved: {OUTPUT_CSV}")

    _plot(m_values, results, OUTPUT_PLOT)
    print(f"Saved: {OUTPUT_PLOT}")


def _plot(m_values, results, path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5.5))
    markers = {"Static": "s", "AugOnly": "o", "MaxCount": "^", "Proposed": "D"}
    colors = {"Static": "gray", "AugOnly": "#2196F3", "MaxCount": "#FF9800",
              "Proposed": "#E91E63"}

    for method, perfs in results.items():
        plt.plot(m_values, perfs,
                 marker=markers.get(method, "x"),
                 color=colors.get(method, "black"),
                 linewidth=1.5, markersize=5, label=method)

    plt.xlabel("m  (k = 10)")
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
    print(f"=== (mk: {N_RUNS} runs per point) ===")
    main()
