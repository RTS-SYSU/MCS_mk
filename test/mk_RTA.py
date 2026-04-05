import copy

import pandas as pd
from matplotlib import pyplot as plt

from scheduling.priority_assignment import assign_static_priorities
from scheduling.sched_test import schedulability_test, schedulability_test_AMCrtbWH
from scheduling.task_partitioning import partition_tasks_wfd, partition_only, partition_only_other
from utils.generate_taskset import generate_taskset


def run_schedulability_experiment():
    # --- 实验参数配置 ---
    TEST_TIMES = 100  # 每个利用率点的测试次数
    NUM_CORES = 1  # 核心数
    TASK_NUM = 20 * NUM_CORES  # 任务总数

    # 任务生成参数
    GEN_PARAMS = {
        'total_processor': NUM_CORES,  # 用于 DRS 生成利用率
        'total_task': TASK_NUM,
        'cp': 0.5, 'cf': 2, 'xf': 1.0,
        'max_hyperperiod': 1440, 'k': 10,
        'targetU': 0.8
    }

    # 利用率范围: 0.5 到 0.9 (步长 0.05)
    # range(10, 19) -> 10, 11, ..., 18 -> 0.40, 0.55, ..., 0.90
    u_steps = [1, 2, 3, 4, 5, 6, 7, 8, 9]

    # --- 结果容器 ---
    summary_data = []  # 汇总表: U, Our_Ratio, AMC_Ratio, Our_Yes_AMC_No, AMC_Yes_Our_No

    # 差异详细记录 (用于保存具体不可调度的案例)
    diff_our_yes_amc_no = []
    diff_amc_yes_our_no = []

    print(f"Starting Schedulability Experiment | Cores: {NUM_CORES} | Samples: {TEST_TIMES}")
    print("-" * 100)
    print(f"{'Util':<6} {'Our%':<8} {'AMC%':<8} {'Our>AMC':<8} {'AMC>Our':<8}")

    for step in u_steps:
        #target_u = step * 0.05
        # 更新生成参数
        current_gen_params = GEN_PARAMS.copy()
        current_gen_params['m'] = step

        cnt_our = 0
        cnt_amc = 0
        cnt_diff_our_better = 0
        cnt_diff_amc_better = 0

        for i in range(TEST_TIMES):
            # 1. 生成任务集
            try:
                tasks = generate_taskset(**current_gen_params)
            except ValueError:
                continue  # 生成失败跳过

            assign_static_priorities(tasks)


            # 3. 深度拷贝用于不同测试
            tasks_our = copy.deepcopy(tasks)
            for task in tasks_our:
                task.mk.m=step
                task.mk.x=10-step
                task.mk.k=10
            tasks_amc = copy.deepcopy(tasks)

            # 4. 执行划分与测试
            # 注意：partition_tasks_wfd 会调用传入的 test_func 来判断能否放入核心
            # 如果返回 None，说明无法划分（即不可调度）

            # A. Our Proposal (Drop-aware RTA or improved RTA)
            # schedulability_test 应该是您改进后的算法
            processor_our = partition_only_other(tasks_our, NUM_CORES)
            processor_amc = partition_only_other(tasks_amc, NUM_CORES)
            is_sched_our = False
            is_sched_amc = False

            if processor_our is not None:



                is_sched_our = True
                is_sched_amc = True
                # for p in processor_our:
                #     print(p.tasks)
                for p in processor_our:
                    if not schedulability_test(p.tasks, []):
                        is_sched_our = False
                        break

                for p in processor_amc:
                    if not schedulability_test_AMCrtbWH(p.tasks):
                        is_sched_amc = False
                        break
            # result_our = partition_tasks_wfd(tasks_our, NUM_CORES, schedulability_test)
            # is_sched_our = (result_our is not None)
            #
            # # B. AMC-rtb-WH (Baseline)
            # result_amc = partition_tasks_wfd(tasks_amc, NUM_CORES, schedulability_test_AMCrtbWH)
            # is_sched_amc = (result_amc is not None)

            # 5. 统计
            if is_sched_our: cnt_our += 1
            if is_sched_amc: cnt_amc += 1

            # if is_sched_our and not is_sched_amc:
            #     cnt_diff_our_better += 1
            #     # 记录详细信息 (可选)
            #     # diff_our_yes_amc_no.append({'Util': target_u, 'Seed': i, 'Tasks': str(tasks)})
            #
            # if is_sched_amc and not is_sched_our:
            #     cnt_diff_amc_better += 1
            #     # diff_amc_yes_our_no.append({'Util': target_u, 'Seed': i, 'Tasks': str(tasks)})

        # 计算比率
        ratio_our = cnt_our / TEST_TIMES
        ratio_amc = cnt_amc / TEST_TIMES
        ratio_diff1 = cnt_diff_our_better / TEST_TIMES
        ratio_diff2 = cnt_diff_amc_better / TEST_TIMES

        summary_data.append({
            "m": step,
            "Our_Ratio": ratio_our,
            "AMC_Ratio": ratio_amc,
            "Our_Yes_AMC_No": ratio_diff1,
            "AMC_Yes_Our_No": ratio_diff2
        })

        print(f"{step:<6.2f} {ratio_our:<8.3f} {ratio_amc:<8.3f} {ratio_diff1:<8.3f} {ratio_diff2:<8.3f}")

    print("-" * 100)

    # --- 保存结果 ---
    df_summary = pd.DataFrame(summary_data)

    # 1. 保存汇总表
    filename_main = f"rta_mk_{NUM_CORES}core"
    df_summary.to_excel(f"{filename_main}.xlsx", index=False)
    print(f"Summary saved to {filename_main}.xlsx")

    # 2. 保存差异表 (如果需要)
    # df_diff1 = pd.DataFrame(diff_our_yes_amc_no)
    # df_diff1.to_excel("diff_our_yes_amc_no.xlsx", index=False)

    # df_diff2 = pd.DataFrame(diff_amc_yes_our_no)
    # df_diff2.to_excel("diff_amc_yes_our_no.xlsx", index=False)

    # --- 绘图 ---
    plot_schedulability(df_summary, filename_main, NUM_CORES)


def plot_schedulability(df, filename, num_cores):
    plt.figure(figsize=(10, 6))

    # Plot Our Method
    plt.plot(df["m"], df["Our_Ratio"],
             color='#d62728', marker='o', linestyle='-', linewidth=2, markersize=8,
             label='Our Proposal')

    # Plot AMC Method
    plt.plot(df["m"], df["AMC_Ratio"],
             color='#1f77b4', marker='s', linestyle='--', linewidth=2, markersize=8,
             label='AMC-rtb-WH')

    # 可选：绘制差异 (Diff) - 通常不需要画在主图上，表格体现即可
    # plt.bar(df["Utilization"], df["Our_Yes_AMC_No"], width=0.02, color='green', alpha=0.3, label='Our>AMC')

    #plt.title(f'Schedulability Ratio vs. Utilization ({num_cores} Cores)', fontsize=16)
    plt.xlabel(r'$\frac{m}{k}$', fontsize=14)
    plt.ylabel('Schedulability Ratio', fontsize=14)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.savefig(f"{filename}.png", dpi=300)
    print(f"Plot saved to {filename}.png")
    # plt.show()


if __name__ == "__main__":
    run_schedulability_experiment()
