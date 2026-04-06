import copy
import logging
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from typing import List, Dict, Any, Tuple, Callable

from matplotlib.ticker import MultipleLocator

from scheduling.normalization_utility import calculate_hyperperiod, get_normalization_utility_offline_stats
from scheduling.offline_simulator import uaswc_offline_multicore, cost_utility_density, cost_dynamic_fairness, \
    cost_high_util, cost_low_util
from scheduling.online_simulator import run_multicore_simulation
from scheduling.priority_assignment import assign_static_priorities
from scheduling.sched_test import schedulability_test
from scheduling.task_partitioning import partition_only

from utils.generate_taskset import generate_taskset

logging.basicConfig(level=logging.WARNING)


def _worker_run(args):
    # Same worker logic
    gen_params = args['gen_params']
    num_cores = args['num_cores']
    strategies = args['strategies']

    try:
        tasks = generate_taskset(**gen_params)
    except ValueError:
        return None

    hp = 10*calculate_hyperperiod(tasks)
    if hp == 0: return None
    _, base = get_normalization_utility_offline_stats(tasks, [], 'LO', hp)
    if base == 0: return None
    res = {}

    tasks_m1 = copy.deepcopy(tasks)
    processors_base = partition_only(tasks_m1, num_cores)
    if processors_base is not None:
        for p in processors_base:
            assign_static_priorities(p.tasks)
            if not schedulability_test(p.tasks):
                return None

        lo_executed_sum = 0.0
        hi_executed_sum = 0.0

        for p in processors_base:
            e_lo, total_lo = get_normalization_utility_offline_stats(p.tasks, p.drop_list, mode='LO',
                                                                     grid_hyperperiod=hp)
            e_hi, total_hi = get_normalization_utility_offline_stats(p.tasks, p.drop_list, mode='HI',
                                                                     grid_hyperperiod=hp)
            lo_executed_sum += e_lo
            hi_executed_sum += e_hi
        res["baseline"] = lo_executed_sum / base
        res["baseline_HI"] = hi_executed_sum / base

        sim_pure_online_lo = run_multicore_simulation(
            processors_base, hp, mode_switch_time=float('inf')
        )
        sim_pure_online_hi = run_multicore_simulation(
            processors_base, hp, mode_switch_time=0
        )

        res["pure_online"] = sim_pure_online_lo["utility"]
        res["pure_online_hi"] = sim_pure_online_hi["utility"]

    for name, (func, dyn) in strategies.items():
        tasks_cp = copy.deepcopy(tasks)
        is_feas, procs = uaswc_offline_multicore(tasks_cp, num_cores, func, dyn)
        if not is_feas: return None

        lo_off_sum = sum(get_normalization_utility_offline_stats(p.tasks, p.drop_list, 'LO', hp)[0] for p in procs)
        sim_lo = run_multicore_simulation(procs, hp, float('inf'))
        res[f"{name}_LO_Off"] = lo_off_sum / base
        res[f"{name}_LO_On"] = sim_lo["utility"]

        hi_off_sum = sum(get_normalization_utility_offline_stats(p.tasks, p.drop_list, 'HI', hp)[0] for p in procs)
        sim_hi = run_multicore_simulation(procs, hp, 0.0)
        res[f"{name}_HI_Off"] = hi_off_sum / base
        res[f"{name}_HI_On"] = sim_hi["utility"]

    return res


def run_exp():
    STRATEGIES = {
        "UASWC": (cost_utility_density, False),
        "Fair": (cost_dynamic_fairness, True),
        "HUF": (cost_high_util, False),
        "LUF": (cost_low_util, False),
    }

    TEST_TIMES = 10000
    BASE_CORES = 4

    print(">>> Running m/k Ratio Experiment...")
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    data = []

    pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 3))

    for r in ratios:
        # Fixed k=10, vary m
        m_val = int(r * 10)
        if m_val < 1: m_val = 1

        params = {
            'targetU': 0.65, 'total_processor': BASE_CORES, 'total_task': 20 * BASE_CORES,
            'cp': 0.5, 'cf': 2, 'xf': 1.0, 'max_hyperperiod': 1440,
            'm': m_val, 'k': 10
        }

        args = [{'gen_params': params, 'num_cores': BASE_CORES, 'strategies': STRATEGIES} for _ in range(TEST_TIMES)]
        results = pool.map(_worker_run, args)
        valid = [r for r in results if r is not None]

        if not valid:
            print(f"Ratio: {r} | No valid samples!")
            continue

        avg = {}
        for k in valid[0].keys():
            avg[k] = sum(v[k] for v in valid) / len(valid)

        row = {"X_Value": r}
        row.update(avg)
        data.append(row)
        print(f"Ratio: {r}|valid: {len(valid)}")

    pool.close()

    df = pd.DataFrame(data)
    df.to_excel("secB_result_mk.xlsx", index=False)

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    for i, name in enumerate(STRATEGIES):
        c, m = colors[i], markers[i]
        if f"{name}_LO_On" in df:
            ax1.plot(df["X_Value"], df[f"{name}_LO_On"], c=c, ls='-', marker=m, label=f"{name} (On)")
            ax1.plot(df["X_Value"], df[f"{name}_LO_Off"], c=c, ls='--', marker=m, alpha=0.5, label=f"{name} (Off)")
            ax2.plot(df["X_Value"], df[f"{name}_HI_On"], c=c, ls='-', marker=m, label=f"{name} (On)")
            ax2.plot(df["X_Value"], df[f"{name}_HI_Off"], c=c, ls='--', marker=m, alpha=0.5, label=f"{name} (Off)")

    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(ncol=2, fontsize=9)  # 双列图例
    ax1.xaxis.set_major_locator(MultipleLocator(0.05))
    ax1.yaxis.set_major_locator(MultipleLocator(0.05))
    ax1.set_title("LO Mode vs m/k Ratio")
    ax1.set_xlabel("m/k")
    ax1.legend()
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(ncol=2, fontsize=9)  # 双列图例
    ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    ax2.yaxis.set_major_locator(MultipleLocator(0.05))
    ax2.set_title("HI Mode vs m/k Ratio")
    ax2.set_xlabel("m/k")
    ax2.legend()
    plt.savefig("plot_sens_mk.png")


if __name__ == "__main__":
    run_exp()
