import matplotlib.pyplot as plt
import pandas as pd

def plot_comparison(df, filename):
    plt.figure(figsize=(10, 5))

    # Base1
    plt.plot(df["Utilization"], df["m<k_utility"],
             marker='o', linestyle='-', linewidth=2, markersize=8,
             label='m<k')

    # Base2
    plt.plot(df["Utilization"], df["m=k_utility"],
             marker='s', linestyle='--', linewidth=2, markersize=8,
             label='m=k')

    # Proposed
    plt.plot(df["Utilization"], df["proposed_utility"],
             marker='^', linestyle='-', linewidth=2, markersize=8,color='red',
             label='Proposed')

    # 标题 & 坐标轴
    #plt.title('Performance Comparison vs. m', fontsize=16)
    plt.xlabel(r'$U^{LO}$', fontsize=22)
    plt.ylabel('Normalized Utility', fontsize=22)

    # 范围控制（和你之前风格一致）
    plt.ylim(-0.01, 1.01)

    # 网格 & 图例
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", dpi=300)
    print(f"Plot saved to {filename}.pdf")
    # plt.show()

df = pd.read_excel('secA_result_ulo.xlsx')

plot_comparison(df, "secA_result_ulo_utility")