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
        test_times: int,
        exp_mode: str = 'LO'  # 'LO' 或 'HI'
) -> Dict[str, float]:
    # 初始化累加器
    agg_results = {}
    for s_name in strategies:
        agg_results[f"{s_name}_Off"] = 0.0
        agg_results[f"{s_name}_On"] = 0.0

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

        # 3. 计算分母 (Baseline Total Value)
        # 基于原始任务集，计算 LO 模式下的理想总价值
        # 这是所有 Utility 计算的唯一基准分母
        _, base_total_value = get_normalization_utility_offline_stats(
            tasks, [], mode='LO', grid_hyperperiod=global_hyperperiod
        )

        if base_total_value == 0: continue

        # 4. 运行各策略
        switch_time = 0.0 if exp_mode == 'HI' else float('inf')

        for name, (cost_func, is_dyn) in strategies.items():
            tasks_copy = copy.deepcopy(tasks)

            # --- A. Offline Phase ---
            is_feasible, optimized_processors = uaswc_offline_multicore(
                original_tasks=tasks_copy,
                num_processors=num_cores,
                cost_func=cost_func,
                is_dynamic_strategy=is_dyn
            )

            if not is_feasible:
                # 策略不可行 (Utility = 0)
                continue

                # 计算 Offline Utility 分子
            off_exec_sum = 0.0
            for p in optimized_processors:
                # 传入 exp_mode，正确处理 Backup/Drop 的贡献
                e, _ = get_normalization_utility_offline_stats(
                    p.tasks, p.drop_list, mode=exp_mode, grid_hyperperiod=global_hyperperiod
                )
                off_exec_sum += e

            agg_results[f"{name}_Off"] += (off_exec_sum / base_total_value)

            # --- B. Online Phase ---
            online_stats = run_multicore_simulation(
                processors=optimized_processors,
                duration=global_hyperperiod,
                mode_switch_time=switch_time
            )

            # 计算 Online Utility
            exc_utility = online_stats["utility"]
            agg_results[f"{name}_On"] += exc_utility

        valid_samples += 1

    # 取平均
    if valid_samples > 0:
        for k in agg_results:
            agg_results[k] /= valid_samples

    return agg_results, valid_samples


def plot_results(df, strategies, filename, mode):
    plt.figure(figsize=(12, 7))

    # 颜色和标记库
    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']  # Red, Blue, Green, Orange, Purple
    markers = ['o', 's', '^', 'D', 'v']

    strat_names = list(strategies.keys())

    for i, name in enumerate(strat_names):
        col_on = f"{name}_On"
        col_off = f"{name}_Off"

        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        # 绘制 Online (实线 + 标记)
        if col_on in df.columns:
            plt.plot(df["Utilization"], df[col_on],
                     color=c, linestyle='-', marker=m, markersize=6, label=f"{name} (On)")

        # 绘制 Offline (虚线 + 空心标记/无标记)
        if col_off in df.columns:
            plt.plot(df["Utilization"], df[col_off],
                     color=c, linestyle='--', marker=m, markersize=4, alpha=0.6, label=f"{name} (Off)")

    plt.title(f"Normalized Utility vs. Utilization ({mode} Mode)", fontsize=16)
    plt.xlabel("Average Per-Core Utilization ($U_{avg}$)", fontsize=14)
    plt.ylabel("Normalized Utility", fontsize=14)
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(fontsize=10, loc='best', ncol=2)  # 分两列显示图例

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    print(f"Plot saved to {filename}.png")


def main():
    # ==========================
    # 实验配置
    # ==========================
    NUM_CORES = 4
    TEST_TIMES = 5000

    # 切换模式: 'LO' 或 'HI'
    # 建议跑两次，生成两张图
    EXPERIMENT_MODE = 'HI'

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

    u_values = [x * 0.05 for x in range(8, 17)]  # 0.4 - 0.9

    final_data = []

    print(f"Starting Experiment | Mode: {EXPERIMENT_MODE} | Cores: {NUM_CORES}")
    print("-" * 100)

    # 动态打印表头
    header = f"{'Util':<6} {'Valid':<6}"
    for s in STRATEGIES:
        header += f" | {s[:6]}_Off {s[:6]}_On"
    print(header)
    print("-" * 100)

    # ==========================
    # 主循环
    # ==========================
    for u in u_values:
        res, count = run_single_utilization_point(
            target_util=u,
            num_cores=NUM_CORES,
            fixed_params=FIXED_PARAMS,
            strategies=STRATEGIES,
            test_times=TEST_TIMES,
            exp_mode=EXPERIMENT_MODE
        )

        row = {"Utilization": u}
        log_str = f"{u:<6.2f} {count:<6}"

        for name in STRATEGIES:
            row[f"{name}_Off"] = res[f"{name}_Off"]
            row[f"{name}_On"] = res[f"{name}_On"]
            log_str += f" | {res[f'{name}_Off']:.3f}   {res[f'{name}_On']:.3f} "

        final_data.append(row)
        print(log_str)

    # ==========================
    # 保存与绘图
    # ==========================
    df = pd.DataFrame(final_data)
    filename = f"result_{EXPERIMENT_MODE}_{NUM_CORES}core"
    df.to_excel(f"{filename}.xlsx", index=False)
    print(f"\nData saved to {filename}.xlsx")

    plot_results(df, STRATEGIES, filename, EXPERIMENT_MODE)


if __name__ == "__main__":
    main()
