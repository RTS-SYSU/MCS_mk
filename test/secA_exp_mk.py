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

    # =============================================
    # 方法 1: Fixed (m<k) - 使用原始 (m,k) 不优化
    # =============================================
    tasks_m1 = copy.deepcopy(tasks)
    processors_mk = partition_only(tasks_m1, num_cores)

    if processors_mk is not None:
        mk_sched = True
        for p in processors_mk:
            assign_static_priorities(p.tasks)
            if not schedulability_test(p.tasks):
                mk_sched = False
                break

        if mk_sched:
            lo_sum = 0.0
            hi_sum = 0.0
            for p in processors_mk:
                e_lo, _ = get_normalization_utility_offline_stats(
                    p.tasks, p.drop_list, mode='LO', grid_hyperperiod=hp
                )
                e_hi, _ = get_normalization_utility_offline_stats(
                    p.tasks, p.drop_list, mode='HI', grid_hyperperiod=hp
                )
                lo_sum += e_lo
                hi_sum += e_hi

            res["m<k_sched"] = 1
            res["m<k_utility_LO"] = lo_sum / base
            res["m<k_utility_HI"] = hi_sum / base
        else:
            res["m<k_sched"] = 0
            res["m<k_utility_LO"] = 0.0
            res["m<k_utility_HI"] = 0.0
    else:
        res["m<k_sched"] = 0
        res["m<k_utility_LO"] = 0.0
        res["m<k_utility_HI"] = 0.0

    # =============================================
    # 方法 2: Fixed (m=k) - 所有 job 都是 mandatory
    # =============================================
    tasks_m2 = copy.deepcopy(tasks)
    for t in tasks_m2:
        if t.criticality == 'LO' and t.mk:
            t.mk.m = t.mk.k  # m = k，所有 job mandatory

    processors_meqk = partition_only(tasks_m2, num_cores)

    if processors_meqk is not None:
        meqk_sched = True
        for p in processors_meqk:
            assign_static_priorities(p.tasks)
            if not schedulability_test(p.tasks):
                meqk_sched = False
                break

        if meqk_sched:
            lo_sum = 0.0
            hi_sum = 0.0
            for p in processors_meqk:
                e_lo, _ = get_normalization_utility_offline_stats(
                    p.tasks, p.drop_list, mode='LO', grid_hyperperiod=hp
                )
                e_hi, _ = get_normalization_utility_offline_stats(
                    p.tasks, p.drop_list, mode='HI', grid_hyperperiod=hp
                )
                lo_sum += e_lo
                hi_sum += e_hi

            res["m=k_sched"] = 1
            res["m=k_utility_LO"] = lo_sum / base
            res["m=k_utility_HI"] = hi_sum / base
        else:
            res["m=k_sched"] = 0
            res["m=k_utility_LO"] = 0.0
            res["m=k_utility_HI"] = 0.0
    else:
        res["m=k_sched"] = 0
        res["m=k_utility_LO"] = 0.0
        res["m=k_utility_HI"] = 0.0

    for name, (func, dyn) in strategies.items():
        tasks_cp = copy.deepcopy(tasks)
        is_feas, procs = uaswc_offline_multicore(tasks_cp, num_cores, func, dyn)
        if not is_feas:
            res[f"{name}_sched"] = 0
            res[f"{name}_LO_Off"] = 0.0
            res[f"{name}_LO_On"] = 0.0
            res[f"{name}_HI_Off"] = 0.0
            res[f"{name}_HI_On"] = 0.0
            continue
        res[f"{name}_sched"] = 1

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
        # "Fair": (cost_dynamic_fairness, True),
        # "HUF": (cost_high_util, False),
        # "LUF": (cost_low_util, False),
    }

    TEST_TIMES = 10000
    BASE_CORES = 4

    print(">>> Running m/k Ratio Experiment...")
    ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    data = []

    pool = multiprocessing.Pool(processes=max(1, 50))

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
            #avg[k] = sum(v[k] for v in valid) / len(valid)
            vals = [v[k] for v in valid if k in v]
            avg[k] = sum(vals) / len(vals) if vals else 0.0


        row = {"X_Value": r}
        row.update(avg)
        data.append(row)
        mk_str = f"m<k: {avg.get('m<k_sched', 0):.3f}/{avg.get('m<k_utility_LO', 0):.3f}"
        meqk_str = f"m=k: {avg.get('m=k_sched', 0):.3f}/{avg.get('m=k_utility_LO', 0):.3f}"
        uaswc_str = f"UASWC: {avg.get('UASWC_sched', 0):.3f}/{avg.get('UASWC_LO_Off', 0):.3f}"
        print(f"Ratio: {r} | valid: {len(valid)} | {mk_str} | {meqk_str} | {uaswc_str}")

    pool.close()

    df = pd.DataFrame(data)
    df.to_excel(f"secA_result_mk_{BASE_CORES}core_full.xlsx", index=False)

    # # Plot
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    # colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    # markers = ['o', 's', '^', 'D', 'v']
    #
    # for i, name in enumerate(STRATEGIES):
    #     c, m = colors[i], markers[i]
    #     if f"{name}_LO_On" in df:
    #         ax1.plot(df["X_Value"], df[f"{name}_LO_On"], c=c, ls='-', marker=m, label=f"{name} (On)")
    #         ax1.plot(df["X_Value"], df[f"{name}_LO_Off"], c=c, ls='--', marker=m, alpha=0.5, label=f"{name} (Off)")
    #         ax2.plot(df["X_Value"], df[f"{name}_HI_On"], c=c, ls='-', marker=m, label=f"{name} (On)")
    #         ax2.plot(df["X_Value"], df[f"{name}_HI_Off"], c=c, ls='--', marker=m, alpha=0.5, label=f"{name} (Off)")
    #
    # ax1.grid(True, linestyle='--', alpha=0.5)
    # ax1.legend(ncol=2, fontsize=9)  # 双列图例
    # ax1.xaxis.set_major_locator(MultipleLocator(0.05))
    # ax1.yaxis.set_major_locator(MultipleLocator(0.05))
    # ax1.set_title("LO Mode vs m/k Ratio")
    # ax1.set_xlabel("m/k")
    # ax1.legend()
    # ax2.grid(True, linestyle='--', alpha=0.5)
    # ax2.legend(ncol=2, fontsize=9)  # 双列图例
    # ax2.xaxis.set_major_locator(MultipleLocator(0.1))
    # ax2.yaxis.set_major_locator(MultipleLocator(0.05))
    # ax2.set_title("HI Mode vs m/k Ratio")
    # ax2.set_xlabel("m/k")
    # ax2.legend()
    # plt.savefig("plot_sens_mk.png")


if __name__ == "__main__":
    run_exp()
