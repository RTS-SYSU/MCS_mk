import copy
import logging
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from typing import List, Dict, Any, Tuple, Callable, Optional

from matplotlib.ticker import MultipleLocator

# --- 引入项目模块 ---
from scheduling.normalization_utility import calculate_hyperperiod, get_normalization_utility_offline_stats
from scheduling.offline_simulator import uaswc_offline_multicore, cost_utility_density, cost_dynamic_fairness, \
    cost_high_util, cost_low_util
from scheduling.online_simulator import run_multicore_simulation
from scheduling.priority_assignment import assign_static_priorities
from scheduling.sched_test import schedulability_test
from scheduling.task_partitioning import partition_only

from utils.generate_taskset import generate_taskset

logging.basicConfig(level=logging.WARNING)


# =========================================================================
# Worker: 处理单个任务集的完整流程 (生成 -> 策略评估)
# =========================================================================
def _worker_unified_run(args: Dict[str, Any]) -> Optional[Dict[str, float]]:
    gen_params = args['gen_params']
    num_cores = args['num_cores']
    strategies = args['strategies']

    # 1. 生成任务集
    try:
        tasks = generate_taskset(**gen_params)
    except ValueError:
        return None

    # 2. 确定全局统一时间基准
    global_hyperperiod = calculate_hyperperiod(tasks)  # k*hp, maxi
    if global_hyperperiod == 0: return None

    # 3. 计算统一分母 (Baseline Total Value)
    _, base_total_value = get_normalization_utility_offline_stats(
        tasks, [], mode='LO', grid_hyperperiod=global_hyperperiod
    )
    if base_total_value == 0: return None

    # 结果容器
    result = {}


    #   m<k
    tasks_m1 = copy.deepcopy(tasks)
    processors_base = partition_only(tasks_m1, num_cores)
    if processors_base is None:
        result["m<k_sched"] = 0
        result["m<k_utility"] = 0
        result["m<k_utility_HI"] = 0
    else:
        is_sched = True
        for p in processors_base:
            assign_static_priorities(p.tasks)
            if not schedulability_test(p.tasks):
                is_sched = False
                break

        if not is_sched:
            result["m<k_sched"] = 0
            result["m<k_utility"] = 0
            result["m<k_utility_HI"] = 0
        else:
            lo_executed_sum = 0.0
            hi_executed_sum = 0.0

            for p in processors_base:
                e_lo, total_lo = get_normalization_utility_offline_stats(p.tasks, p.drop_list, mode='LO',
                                                                         grid_hyperperiod=global_hyperperiod)
                e_hi, total_hi = get_normalization_utility_offline_stats(p.tasks, p.drop_list, mode='HI',
                                                                         grid_hyperperiod=global_hyperperiod)
                lo_executed_sum += e_lo
                hi_executed_sum += e_hi
            result["m<k_sched"] = 1
            result["m<k_utility"] = lo_executed_sum / base_total_value
            result["m<k_utility_HI"] = hi_executed_sum / base_total_value


    #  m=k
    tasks_m2 = copy.deepcopy(tasks)
    for t in tasks_m2:
        if t.criticality == 'LO' and t.mk:
            original_k = t.mk.k
            t.mk.m = original_k
    processors_m2 = partition_only(tasks_m2, num_cores)
    if processors_m2 is None:
        result["m=k_sched"] = 0
        result["m=k_utility"] = 0.0
        result["m=k_utility_HI"] = 0.0
    else:
        is_sched = True
        for p in processors_m2:
            assign_static_priorities(p.tasks)
            if not schedulability_test(p.tasks):
                is_sched = False
                break

        if not is_sched:
            result["m=k_sched"] = 0
            result["m=k_utility"] = 0.0
            result["m=k_utility_HI"] = 0.0
        else:
            lo_executed_sum = 0.0
            hi_executed_sum = 0.0
            for p in processors_m2:
                e_lo, _ = get_normalization_utility_offline_stats(
                    p.tasks, p.drop_list, mode='LO',
                    grid_hyperperiod=global_hyperperiod
                )
                e_hi, _ = get_normalization_utility_offline_stats(
                    p.tasks, p.drop_list, mode='HI',
                    grid_hyperperiod=global_hyperperiod
                )
                lo_executed_sum += e_lo
                hi_executed_sum += e_hi

            result["m=k_sched"] = 1
            result["m=k_utility"] = lo_executed_sum / base_total_value
            result["m=k_utility_HI"] = hi_executed_sum / base_total_value

    # 4. 计算各优化策略
    for name, (cost_func, is_dyn) in strategies.items():
        tasks_strategy = copy.deepcopy(tasks)

        # 4.1 离线优化
        is_feasible, optimized_processors = uaswc_offline_multicore(
            original_tasks=tasks_strategy,
            num_processors=num_cores,
            cost_func=cost_func,
            is_dynamic_strategy=is_dyn
        )

        if not is_feasible:
            # all_strategies_feasible = False
            result[f"{name}_sched"] = 0
            result[f"{name}_LO_Off"] = 0
            result[f"{name}_LO_On"] = 0
            result[f"{name}_HI_Off"] = 0
            result[f"{name}_HI_On"] = 0
            continue

        result[f"{name}_sched"] = 1
        # 4.2 LO 模式评估
        lo_off_sum = 0.0
        for p in optimized_processors:
            e, _ = get_normalization_utility_offline_stats(
                p.tasks, p.drop_list, mode='LO', grid_hyperperiod=global_hyperperiod
            )
            lo_off_sum += e
        result[f"{name}_LO_Off"] = lo_off_sum / base_total_value

        sim_lo = run_multicore_simulation(
            optimized_processors, global_hyperperiod, mode_switch_time=float('inf')
        )

        result[f"{name}_LO_On"] = sim_lo["utility"]

        # 4.3 HI 模式评估
        hi_off_sum = 0.0
        for p in optimized_processors:
            e, _ = get_normalization_utility_offline_stats(
                p.tasks, p.drop_list, mode='HI', grid_hyperperiod=global_hyperperiod
            )
            hi_off_sum += e
        result[f"{name}_HI_Off"] = hi_off_sum / base_total_value

        sim_hi = run_multicore_simulation(
            optimized_processors, global_hyperperiod, mode_switch_time=0.0
        )
        result[f"{name}_HI_On"] = sim_hi["utility"]

    return result


# =========================================================================
# Parallel Manager
# =========================================================================
def run_single_utilization_point_parallel(
        target_util: float,
        num_cores: int,
        fixed_params: Dict[str, Any],
        strategies: Dict[str, Tuple[Callable, bool]],
        test_times: int
) -> Tuple[Dict[str, float], int]:
    # 准备参数
    current_gen_params = fixed_params.copy()
    current_gen_params['targetU'] = target_util
    current_gen_params['total_processor'] = num_cores

    # 创建任务列表
    # 注意：Pool.map 需要 picklable 的参数。Strategies 中的 Callable (函数) 是 picklable 的。
    batch_args = [{
        'gen_params': current_gen_params,
        'num_cores': num_cores,
        'strategies': strategies
    } for _ in range(test_times)]

    # 启动进程池
    num_processes = max(1, 50)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_list = pool.map(_worker_unified_run, batch_args)

    # 聚合结果
    valid_results = [r for r in results_list if r is not None]
    valid_count = len(valid_results)

    # 初始化聚合字典
    agg_results = {}

    agg_results["m<k_sched"] = 0
    agg_results["m<k_utility"] = 0
    agg_results["m<k_utility_HI"] = 0
    agg_results["m=k_sched"] = 0
    agg_results["m=k_utility"] = 0
    agg_results["m=k_utility_HI"] = 0
    for s in strategies:
        agg_results[f"{s}_LO_Off"] = 0.0
        agg_results[f"{s}_LO_On"] = 0.0
        agg_results[f"{s}_HI_Off"] = 0.0
        agg_results[f"{s}_HI_On"] = 0.0
        agg_results[f"{s}_sched"] = 0.0

    if valid_count > 0:
        for r in valid_results:
            for k, v in r.items():
                agg_results[k] += v

        # 取平均
        for k in agg_results:
            agg_results[k] /= valid_count

    return agg_results, valid_count


# =========================================================================
# 绘图与主函数
# =========================================================================
# def plot_mode_results(df, strategies, mode, filename):
#     plt.figure(figsize=(9, 6))
#
#     colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e']
#     markers = ['o', 's', '^', 'D']
#
#     for i, name in enumerate(strategies):
#         c = colors[i % len(colors)]
#         m = markers[i % len(markers)]
#
#         on_col = f"{name}_{mode}_On"
#         off_col = f"{name}_{mode}_Off"
#
#         if on_col in df.columns:
#             plt.plot(df["Utilization"], df[on_col],
#                      color=c, marker=m, linestyle='-',
#                      label=f"{name} (Online)")
#
#         if off_col in df.columns:
#             plt.plot(df["Utilization"], df[off_col],
#                      color=c, marker=m, linestyle='--', alpha=0.6,
#                      label=f"{name} (Offline)")
#
#     plt.xlabel(r"$U_{avg}^{LO}$")
#     plt.ylabel("Normalized Utility")
#     plt.title(f"{mode} Mode Performance")
#     plt.grid(True, linestyle='--', alpha=0.5)
#     plt.legend(ncol=2, fontsize=10)
#
#     plt.tight_layout()
#     plt.savefig(f"{filename}.pdf")
#     plt.close()
#
#     print(f"{mode} plot saved to {filename}.pdf")


def export_mode_tables(df, strategies, base_filename):
    lo_cols = ["Utilization"]
    hi_cols = ["Utilization"]

    for s in strategies:
        lo_cols += [f"{s}_LO_Off", f"{s}_LO_On"]
        hi_cols += [f"{s}_HI_Off", f"{s}_HI_On"]

    df_lo = df[lo_cols]
    df_hi = df[hi_cols]

    df_lo.to_excel(f"{base_filename}_LO_table.xlsx", index=False)
    df_hi.to_excel(f"{base_filename}_HI_table.xlsx", index=False)

    print("Tables exported:")
    print(f" - {base_filename}_LO_table.xlsx")
    print(f" - {base_filename}_HI_table.xlsx")


def main():
    NUM_CORES = 4
    TEST_TIMES = 100  # 并行后可以跑大样本

    STRATEGIES = {
        "UASWC": (cost_utility_density, False),
        # "Fair": (cost_dynamic_fairness, True),
        # "HUF": (cost_high_util, False),
        # "LUF": (cost_low_util, False),
    }

    FIXED_PARAMS = {
        'total_task': 20 * NUM_CORES,
        'cp': 0.5, 'cf': 2, 'xf': 1.0,
        'max_hyperperiod': 1440,
        'm': None, 'k': None
    }

    # u_values = [x * 0.05 for x in range(10, 19)]
    u_values = [x * 0.05 for x in range(8, 19)]
    final_data = []

    print(f"Starting Parallel Experiment | Cores: {NUM_CORES} | Samples: {TEST_TIMES}")
    print("-" * 100)
    print(f"{'Util':<6} {'Valid':<6} | UASWC (LO: Off/On | HI: Off/On)...")

    for u in u_values:
        # 调用并行版运行函数
        res, count = run_single_utilization_point_parallel(
            target_util=u,
            num_cores=NUM_CORES,
            fixed_params=FIXED_PARAMS,
            strategies=STRATEGIES,
            test_times=TEST_TIMES
        )

        row = {"Utilization": u}
        for k, v in res.items():
            row[k] = v
        final_data.append(row)

        lo_str = f"{res['UASWC_LO_Off']:.2f}/{res['UASWC_LO_On']:.2f}"
        hi_str = f"{res['UASWC_HI_Off']:.2f}/{res['UASWC_HI_On']:.2f}"
        print(f"{u:<6.2f} {count:<6} | {lo_str:<11} | {hi_str:<11}")

    df = pd.DataFrame(final_data)
    filename = f"secA_result_Ulo_{NUM_CORES}core"

    # 总表（可选）
    df.to_excel(f"{filename}_full.xlsx", index=False)

    # 导出 LO / HI 表
    export_mode_tables(df, STRATEGIES, filename)

    # 绘制 LO / HI 图（PDF）
    # plot_mode_results(df, STRATEGIES, mode="LO", filename=f"{filename}_LO")
    # plot_mode_results(df, STRATEGIES, mode="HI", filename=f"{filename}_HI")


if __name__ == "__main__":
    main()
