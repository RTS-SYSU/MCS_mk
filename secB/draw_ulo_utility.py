import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. 读取 Excel 数据 (修改为了 read_excel)
# 确保你已经安装了 openpyxl： pip install openpyxl
df = pd.read_excel('secB_result_mk.xlsx')

# 定义策略和颜色 (Off 用深色，On 用浅色或带斜线)
# strategies = ['UASWC', 'LUF','Fair','HUF']
# strategies = ['HUF']
colors_on = ['#30617F', '#E8977A', '#AED9E6', '#CCB89E']  # 你提供的基准颜色
colors_off = ['#97B0BF', '#F3CBBC', '#D6ECF2', '#E5DBCE']  # 精准计算的对应浅色

# 2. 创建画布 (1行2列)
# fig, axes = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
# fig.suptitle('Performance Evaluation: LO Mode vs HI Mode', fontsize=16)

x = np.arange(len(df['Utilization']))  # x轴刻度位置
width = 0.3  # 每个柱子的宽度
# 8个柱子的偏移量计算 (中心对称)
offsets = np.linspace(-1 * width, 1 * width, 3)


# 3. 定义绘图函数 (复用逻辑)
def plot_mode(mode_str):
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_plotted_off = []
    bars_plotted_on = []
    labels_off = []
    labels_on = []

    # 画柱状图
    idx = 0
    bar1 = ax.bar(x + offsets[idx], df[f'pure_online_{mode_str}'], width, color='#D6ECF2', edgecolor='black', linewidth=0.8,
                  hatch='..')
    bars_plotted_on.append(bar1)
    labels_on.append('Proposed_On')
    idx += 1
    bar1 = ax.bar(x + offsets[idx], df['baseline'], width, color='gray', edgecolor='black', linewidth=0.8)
    bars_plotted_on.append(bar1)
    labels_on.append('Baseline')
    idx += 1



    # for i, strat in enumerate(strategies):
    #     # Online 柱子 (使用浅色并加上斜线阴影)
    #     col_on = f'{strat}_{mode_str}_On'
    #     bar2 = ax.bar(x + offsets[idx], df[col_on], width, color=colors_on[i], edgecolor='black',linewidth=0.8,hatch='..')
    #     bars_plotted_on.append(bar2)
    #     labels_on.append(f'{strat}_On')
    #     idx += 1

    # 画 Baseline 折线图
    # line = ax.plot(x, df['Baseline'], color='black', marker='D', markersize=6,
    #                linestyle='-', linewidth=2, label='Baseline', zorder=10)

    # 设置坐标轴和格式
    # ax.set_title(title, fontsize=14)
    #ax.set_xlabel(r'$U_{avg}^{LO}$', fontsize=22)
    ax.set_xlabel('Utilization', fontsize=22)
    ax.set_ylabel('Normalized Utility', fontsize=22)
    ax.set_xticks(x)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.set_xticklabels([f"{val:.2f}" for val in df['Utilization']])
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # 根据数据范围设置y轴，留出放图例的空间
    ax.set_ylim(0.4, 1.01)
    # ax.set_xlim(0.5, 0.8)

    # 添加图例
    # handles, _ = ax.get_legend_handles_labels()
    s = 0
    all_handles = []
    all_labels = []
    while s < 2:
        all_handles.append(bars_plotted_on[s])
        # all_handles.append(bars_plotted_on[s])
        all_labels.append(labels_on[s])
        # all_labels.append(labels_on[s])
        s += 1
    # all_handles += line
    # all_labels += ['Baseline']
    ax.legend(all_handles, all_labels, bbox_to_anchor=(0.01, 1), loc='lower left', ncol=5, fontsize=14, frameon=True)

    plt.tight_layout()
    # bbox_inches='tight' 极度重要！它保证放在图外的图例在输出 PDF 时不会被裁掉
    plt.savefig(f"result_ulo_utility_on_{mode_str}.pdf", format='pdf', bbox_inches='tight')
    plt.close()  # 必须 close，否则两张图的线条会叠在一起
    print(f"✅ 图表已成功保存为: result_ulo_utility_off_{mode_str}")


# 4. 分别绘制 LO 和 HI
plot_mode('LO')

plot_mode('HI')
