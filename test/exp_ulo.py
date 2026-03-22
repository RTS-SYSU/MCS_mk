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
    global_hyperperiod = calculate_hyperperiod(tasks)
    if global_hyperperiod == 0: return None

    # 3. 计算统一分母 (Baseline Total Value)
    _, base_total_value = get_normalization_utility_offline_stats(
        tasks, [], mode='LO', grid_hyperperiod=global_hyperperiod
    )
    if base_total_value == 0: return None

    # 结果容器
    result = {}
    all_strategies_feasible = True

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
            all_strategies_feasible = False
            break

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

    if all_strategies_feasible:
        return result
    else:
        return None


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
    num_processes = max(1, multiprocessing.cpu_count() - 3)
    with multiprocessing.Pool(processes=num_processes) as pool:
        results_list = pool.map(_worker_unified_run, batch_args)

    # 聚合结果
    valid_results = [r for r in results_list if r is not None]
    valid_count = len(valid_results)

    # 初始化聚合字典
    agg_results = {}
    for s in strategies:
        agg_results[f"{s}_LO_Off"] = 0.0
        agg_results[f"{s}_LO_On"] = 0.0
        agg_results[f"{s}_HI_Off"] = 0.0
        agg_results[f"{s}_HI_On"] = 0.0

    if valid_count > 0:
        for r in valid_results:
            for k, v in r.items():
                agg_results[k] += v

        # 取平均
        for k in agg_results:
            agg_results[k] /= valid_count

    return agg_results, valid_count


# =========================================================================
# 绘图与主函数 (保持原有风格)
# =========================================================================
def plot_combined_results(df, strategies, filename):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    for i, name in enumerate(strategies):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        # LO Mode
        if f"{name}_LO_On" in df.columns:
            ax1.plot(df["Utilization"], df[f"{name}_LO_On"], c=c, ls='-', marker=m, label=f"{name} (On)")
            ax1.plot(df["Utilization"], df[f"{name}_LO_Off"], c=c, ls='--', marker=m, alpha=0.5,label=f"{name} (Off)")  # Off 不加label免得乱

        # HI Mode
        if f"{name}_HI_On" in df.columns:
            ax2.plot(df["Utilization"], df[f"{name}_HI_On"], c=c, ls='-', marker=m, label=f"{name} (On)")
            ax2.plot(df["Utilization"], df[f"{name}_HI_Off"], c=c, ls='--', marker=m, alpha=0.5,label=f"{name} (Off)")

    for ax, title in zip([ax1, ax2], ["LO Mode Performance", "HI Mode Performance"]):
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Utilization ($U_{avg^{LO}}$)")
        ax.set_ylabel("Normalized Utility")
        #ax.set_ylim(0, 1.05)
        ax.grid(True, ls='--', alpha=0.5)
        ax.legend(ncol=2, fontsize=9)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.yaxis.set_major_locator(MultipleLocator(0.05))

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    print(f"Plot saved to {filename}.png")


def main():
    NUM_CORES = 4
    TEST_TIMES = 5000  # 并行后可以跑大样本

    STRATEGIES = {
        "UASWC": (cost_utility_density, False),
        "Fair": (cost_dynamic_fairness, True),
        "HUF": (cost_high_util, False),
        "LUF": (cost_low_util, False),
    }

    FIXED_PARAMS = {
        'total_task': 20 * NUM_CORES,
        'cp': 0.5, 'cf': 2.0, 'xf': 1.0,
        'max_hyperperiod': 1440,
        'm': None, 'k': None
    }

    u_values = [x * 0.05 for x in range(10, 19)]
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
    filename = f"result_parallel_{NUM_CORES}core"
    df.to_excel(f"{filename}.xlsx", index=False)
    print(f"\nData saved to {filename}.xlsx")

    plot_combined_results(df, STRATEGIES, filename)


if __name__ == "__main__":
    main()
