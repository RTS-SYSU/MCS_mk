import pandas as pd
from matplotlib import pyplot as plt


def plot_schedulability(df, filename, num_cores):
    plt.figure(figsize=(10, 5))

    # Plot Our Method
    plt.plot(df["Utilization"], df["Our_Ratio"],
             color='#d62728', marker='o', linestyle='-', linewidth=2, markersize=8,
             label='Proposed')

    # Plot AMC Method
    plt.plot(df["Utilization"], df["AMC_Ratio"],
             color='#1f77b4', marker='s', linestyle='--', linewidth=2, markersize=8,
             label='AMC-rtb-WH')

    # 可选：绘制差异 (Diff) - 通常不需要画在主图上，表格体现即可
    # plt.bar(df["Utilization"], df["Our_Yes_AMC_No"], width=0.02, color='green', alpha=0.3, label='Our>AMC')

    #plt.title(f'Schedulability Ratio vs. Utilization ({num_cores} Cores)', fontsize=16)
    plt.xlabel(r'Utilization', fontsize=22)
    plt.ylabel('Acceptance Ratio', fontsize=22)
    plt.ylim(-0.01, 1.01)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", dpi=300)
    print(f"Plot saved to {filename}.pdf")
    # plt.show()

df = pd.read_excel('schedulability_comparison_4core.xlsx')
plot_schedulability(df,"secC_result_ulo",1)

