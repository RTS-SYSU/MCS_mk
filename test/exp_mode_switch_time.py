import copy
import logging
import pandas as pd
import matplotlib.pyplot as plt
import multiprocessing
from typing import List, Dict, Any, Tuple, Callable, Optional
from matplotlib.ticker import MultipleLocator

# Import Modules
from scheduling.normalization_utility import calculate_hyperperiod, get_normalization_utility_offline_stats
from scheduling.offline_simulator import uaswc_offline_multicore, cost_utility_density, cost_dynamic_fairness, \
    cost_high_util, cost_low_util
from scheduling.online_simulator import run_multicore_simulation

from utils.generate_taskset import generate_taskset

logging.basicConfig(level=logging.WARNING)


def _worker_switch_time_run(args):
    """
    Worker: 给定一个任务集，测试不同 Switch Time 下的性能。
    注意：这里我们不需要在 Worker 内部遍历 Switch Time，而是让 Runner 遍历。
    或者为了效率，Worker 对一个任务集跑完所有 Switch Time 点。
    后者效率更高（任务集只生成一次，offline 只跑一次）。
    """
    gen_params = args['gen_params']
    num_cores = args['num_cores']
    strategies = args['strategies']
    switch_ratios = args['switch_ratios']  # [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    try:
        tasks = generate_taskset(**gen_params)
    except ValueError:
        return None

    hp = calculate_hyperperiod(tasks)
    if hp == 0: return None
    _, base_total = get_normalization_utility_offline_stats(tasks, [], 'LO', hp)
    if base_total == 0: return None

    # 结果字典: { "Ratio_0.0": {Strat1: val, Strat2: val}, "Ratio_0.2": ... }
    # 为了方便聚合，我们返回扁平化结构: { "0.0_UASWC": val, "0.2_UASWC": val ... }
    results = {}
    all_feasible = True

    for name, (cost_func, is_dyn) in strategies.items():
        tasks_cp = copy.deepcopy(tasks)

        # 1. Offline Phase (只跑一次)
        # Offline 优化结果是静态的，它决定了哪些任务被 Drop，哪些有 Backup，x 是多少。
        # 这些决策不依赖于实际的 Switch Time (因为 Offline 必须覆盖最坏情况)。
        is_feas, procs = uaswc_offline_multicore(tasks_cp, num_cores, cost_func, is_dyn)

        if not is_feas:
            all_feasible = False
            break

        # 2. Online Phase (遍历不同的 Switch Time)
        for ratio in switch_ratios:
            # 计算绝对时间
            t_switch = ratio * hp

            # 特殊处理 1.0 (模拟 infinite，永不切换)
            # 虽然 1.0 * hp 也是一个时间点，但为了稳妥，如果是 1.0 就设为 inf
            # 或者就在 hp 结束时切换，效果是一样的（整个 hp 都是 LO）
            if ratio >= 1.0:
                t_switch = float('inf')

            sim_stats = run_multicore_simulation(
                procs,
                duration=hp,  # 仿真时长 = 1个超周期
                mode_switch_time=t_switch
            )

            # 记录该策略在该切换时间点下的 Utility
            key = f"{ratio:.1f}_{name}"
            results[key] = sim_stats["utility"]

    if all_feasible:
        return results
    else:
        return None


def run_exp():
    # 配置
    STRATEGIES = {
        "UASWC": (cost_utility_density, False),
        "Fair": (cost_dynamic_fairness, True),
        "HUF": (cost_high_util, False),
        "LUF": (cost_low_util, False),
    }

    TEST_TIMES = 5000
    BASE_CORES = 4
    SWITCH_RATIOS = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # 固定负载参数 (选择一个有区分度的负载，例如 0.75 或 0.8)
    FIXED_UTIL = 0.75

    print(f">>> Running Mode Switch Time Experiment (U={FIXED_UTIL})...")

    params = {
        'targetU': FIXED_UTIL, 'total_processor': BASE_CORES, 'total_task': 20 * BASE_CORES,
        'cp': 0.5, 'cf': 2.0, 'xf': 1.0, 'max_hyperperiod': 1440, 'm': None, 'k': None
    }

    # 准备参数
    batch_args = [{
        'gen_params': params,
        'num_cores': BASE_CORES,
        'strategies': STRATEGIES,
        'switch_ratios': SWITCH_RATIOS
    } for _ in range(TEST_TIMES)]

    # 并行执行
    pool = multiprocessing.Pool(processes=max(1, multiprocessing.cpu_count() - 3))
    results_list = pool.map(_worker_switch_time_run, batch_args)
    pool.close()

    # 聚合
    valid_results = [r for r in results_list if r is not None]
    if not valid_results:
        print("No valid samples!")
        return

    print(f"Valid Samples: {len(valid_results)}")

    # 重组数据为 DataFrame 格式
    # 目标: [Ratio, UASWC, Fair, HUF, LUF]
    final_data = []

    for r in SWITCH_RATIOS:
        key_prefix = f"{r:.1f}"
        row = {"Switch_Time_Ratio": r}

        for name in STRATEGIES:
            # 提取所有样本中该 Ratio 该策略的值，取平均
            vals = [sample[f"{key_prefix}_{name}"] for sample in valid_results]
            avg_val = sum(vals) / len(vals)
            row[name] = avg_val

        final_data.append(row)
        print(f"Ratio {r}: {row}")

    df = pd.DataFrame(final_data)
    df.to_excel("result_switch_time.xlsx", index=False)

    plot_switch_time(df, STRATEGIES, "plot_switch_time")


def plot_switch_time(df, strategies, filename):
    fig, ax = plt.subplots(figsize=(10, 7))

    colors = ['#d62728', '#1f77b4', '#2ca02c', '#ff7f0e', '#9467bd']
    markers = ['o', 's', '^', 'D', 'v']

    for i, name in enumerate(strategies):
        c = colors[i % len(colors)]
        m = markers[i % len(markers)]

        ax.plot(df["Switch_Time_Ratio"], df[name], c=c, ls='-', marker=m, label=f"{name} (On)")

    ax.set_title("Impact of Mode Switch Time on System Utility", fontsize=16)
    ax.set_xlabel("Mode Switch Time (Ratio of Hyperperiod)", fontsize=14)
    ax.set_ylabel("Normalized Utility", fontsize=14)

    # 设置刻度
    ax.xaxis.set_major_locator(MultipleLocator(0.2))  # X轴间隔 0.1
    ax.yaxis.set_major_locator(MultipleLocator(0.05))  # Y轴间隔 0.05
    # ax.set_ylim(0, 1.05)
    # ax.set_xlim(-0.05, 1.05)

    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(fontsize=12)


    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    print(f"Plot saved to {filename}.png")


if __name__ == "__main__":
    run_exp()
