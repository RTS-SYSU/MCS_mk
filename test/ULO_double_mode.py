import copy
import logging
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Callable

from scheduling.normalization_utility import calculate_hyperperiod, get_normalization_utility_offline_stats
from scheduling.offline_simulator import uaswc_offline_multicore, cost_utility_density, cost_dynamic_fairness, \
    cost_high_util, cost_low_util
from scheduling.online_simulator import run_multicore_simulation

from utils.generate_taskset import generate_taskset

# 配置日志
logging.basicConfig(level=logging.WARNING)


def run_single_utilization_point(
        target_util: float,
        num_cores: int,
        fixed_params: Dict[str, Any],
        strategies: Dict[str, Tuple[Callable, bool]],
        test_times: int
) -> Dict[str, float]:
    """
    针对单个利用率点，生成任务集，并在同一任务集上同时测试 LO 和 HI 模式。
    包含 Offline 和 Online 结果。
    """

    # 初始化累加器
    # 结构: Strategy_Mode_Type (e.g., UASWC_LO_On, UASWC_HI_Off)
    agg_results = {}

    for s in strategies:
        agg_results[f"{s}_LO_Off"] = 0.0
        agg_results[f"{s}_LO_On"] = 0.0
        agg_results[f"{s}_HI_Off"] = 0.0
        agg_results[f"{s}_HI_On"] = 0.0

    valid_samples = 0

    for i in range(test_times):
        # 1. 生成任务集
        current_gen_params = fixed_params.copy()
        current_gen_params['targetU'] = target_util
        current_gen_params['total_processor'] = num_cores

        try:
            tasks = generate_taskset(**current_gen_params)
        except ValueError:
            continue

        # 2. 确定全局统一时间基准
        global_hyperperiod =  calculate_hyperperiod(tasks)
        if global_hyperperiod == 0: continue

        # 3. 计算统一分母 (Baseline Total Value)
        # 所有模式(LO/HI)和所有类型(Off/On)都除以这个值
        _, base_total_value = get_normalization_utility_offline_stats(
            tasks, [], mode='LO', grid_hyperperiod=global_hyperperiod
        )
        if base_total_value == 0: continue

        # ==========================================
        # 4. 计算各优化策略
        # ==========================================
        all_strategies_feasible = True
        for name, (cost_func, is_dyn) in strategies.items():
            # 每次策略必须使用原始任务集的全新副本
            tasks_strategy = copy.deepcopy(tasks)

            # 4.1 离线优化 (只跑一次!)
            # 这个结果包含了分配、x优化、Drop标记、Backup生成

            is_feasible, optimized_processors = uaswc_offline_multicore(
                original_tasks=tasks_strategy,
                num_processors=num_cores,
                cost_func=cost_func,
                is_dynamic_strategy=is_dyn
            )

            if not is_feasible:
                # 如果不可行，该样本该策略得分为 0
                all_strategies_feasible = False
                break

                # ---------------------------
            # 4.2 评估 LO 模式
            # ---------------------------
            # Offline Calculation
            lo_off_sum = 0.0
            for p in optimized_processors:
                e, _ = get_normalization_utility_offline_stats(
                    p.tasks, p.drop_list, mode='LO', grid_hyperperiod=global_hyperperiod
                )
                lo_off_sum += e
            agg_results[f"{name}_LO_Off"] += (lo_off_sum / base_total_value)

            # Online Simulation
            sim_lo = run_multicore_simulation(
                optimized_processors, global_hyperperiod, mode_switch_time=float('inf')
            )
            agg_results[f"{name}_LO_On"] += (sim_lo["utility"])

            # ---------------------------
            # 4.3 评估 HI 模式
            # ---------------------------
            # Offline Calculation
            hi_off_sum = 0.0
            for p in optimized_processors:
                # 传入 mode='HI'，函数会自动计算 Backup 的贡献
                e, _ = get_normalization_utility_offline_stats(
                    p.tasks, p.drop_list, mode='HI', grid_hyperperiod=global_hyperperiod
                )
                hi_off_sum += e
            agg_results[f"{name}_HI_Off"] += (hi_off_sum / base_total_value)

            # Online Simulation (Switch at 0)
            sim_hi = run_multicore_simulation(
                optimized_processors, global_hyperperiod, mode_switch_time=0.0
            )
            agg_results[f"{name}_HI_On"] += sim_hi["utility"]
        if all_strategies_feasible:
            valid_samples += 1


    # 取平均值
    if valid_samples > 0:
        for k in agg_results:
            agg_results[k] /= valid_samples

    return agg_results, valid_samples


def plot_combined_results(df, strategies, filename):
    """
    绘制两张子图：左边 LO 模式，右边 HI 模式。
    包含 Offline (虚线) 和 Online (实线)。
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    # 颜色和标记库
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']
    strat_names = list(strategies.keys())

    # --- Plot 1: LO Mode ---
    for i, name in enumerate(strat_names):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        # Online (Solid)
        if f"{name}_LO_On" in df.columns:
            ax1.plot(df["Utilization"], df[f"{name}_LO_On"],
                     color=c, linestyle='-', marker=m, markersize=4, label=f"{name} (On)")
        # Offline (Dashed)
        if f"{name}_LO_Off" in df.columns:
            ax1.plot(df["Utilization"], df[f"{name}_LO_Off"],
                     color=c, linestyle='--', marker=m, markersize=4, alpha=0.5, label=f"{name} (Off)")

    ax1.set_title("LO Mode Performance", fontsize=14)
    ax1.set_xlabel("Average Utilization ($U_{LO}$)", fontsize=12)
    ax1.set_ylabel("Normalized Utility", fontsize=12)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(ncol=2, fontsize=9)  # 双列图例

    # --- Plot 2: HI Mode ---
    for i, name in enumerate(strat_names):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        # Online (Solid)
        if f"{name}_HI_On" in df.columns:
            ax2.plot(df["Utilization"], df[f"{name}_HI_On"],
                     color=c, linestyle='-', marker=m, markersize=4, label=f"{name} (On)")
        # Offline (Dashed)
        if f"{name}_HI_Off" in df.columns:
            ax2.plot(df["Utilization"], df[f"{name}_HI_Off"],
                     color=c, linestyle='--', marker=m, markersize=4, alpha=0.5, label=f"{name} (Off)")

    ax2.set_title("HI Mode Performance (Fault Tolerance)", fontsize=14)
    ax2.set_xlabel("Average Utilization ($U_{LO}$)", fontsize=12)
    ax2.set_ylabel("Normalized Utility", fontsize=12)
    ax2.set_ylim(None, 1.05)
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(ncol=2, fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    print(f"Plot saved to {filename}.png")


def main():
    # ==========================
    # 实验配置
    # ==========================
    NUM_CORES = 4
    TEST_TIMES = 5000

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

    u_values = [x * 0.05 for x in range(10, 19)]  # 0.4 - 0.9

    final_data = []

    print(f"Starting Unified Experiment | Cores: {NUM_CORES}")
    print("-" * 100)

    # 打印表头
    print(f"{'Util':<6} {'Valid':<6} | UASWC (LO: Off/On | HI: Off/On)...")

    for u in u_values:
        res, count = run_single_utilization_point(
            target_util=u,
            num_cores=NUM_CORES,
            fixed_params=FIXED_PARAMS,
            strategies=STRATEGIES,
            test_times=TEST_TIMES
        )

        # 整理数据
        row = {"Utilization": u}
        for k, v in res.items():
            row[k] = v

        final_data.append(row)

        # 简单打印 UASWC 的结果作为进度指示
        lo_str = f"{res['UASWC_LO_Off']:.2f}/{res['UASWC_LO_On']:.2f}"
        hi_str = f"{res['UASWC_HI_Off']:.2f}/{res['UASWC_HI_On']:.2f}"
        print(f"{u:<6.2f} {count:<6} | {lo_str:<11} | {hi_str:<11}")

    # 保存
    df = pd.DataFrame(final_data)
    filename = f"result_double_{NUM_CORES}core"
    df.to_excel(f"{filename}.xlsx", index=False)
    print(f"\nData saved to {filename}.xlsx")

    # 绘图
    plot_combined_results(df, STRATEGIES, filename)


if __name__ == "__main__":
    main()
