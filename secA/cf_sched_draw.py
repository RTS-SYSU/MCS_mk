import matplotlib.pyplot as plt
import pandas as pd

def plot_comparison(df, filename):
    plt.figure(figsize=(10, 5))

    # Base1
    plt.plot(df["cf"], df["base1"],
             marker='o', linestyle='-', linewidth=2, markersize=8,
             label='m<k')

    # Base2
    plt.plot(df["cf"], df["base2"],
             marker='s', linestyle='--', linewidth=2, markersize=8,
             label='m=k')

    # Proposed
    plt.plot(df["cf"], df["proposed"],
             marker='^', linestyle='-', linewidth=2, markersize=8,color='red',
             label='Proposed')

    # 标题 & 坐标轴
    #plt.title('Performance Comparison vs. m', fontsize=16)
    plt.xlabel('cf', fontsize=22)
    plt.ylabel('Schedulability Ratio', fontsize=22)

    # 范围控制（和你之前风格一致）
    plt.ylim(-0.01, 1.01)

    # 网格 & 图例
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", dpi=300)
    print(f"Plot saved to {filename}.pdf")
    # plt.show()

data = {
    'cf': [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3],
    'base1': [1.0, 1.0, 0.9996, 0.9794, 0.8562, 0.5306, 0.1155, 0.003, 0.0, 0.0, 0.0],
    'base2': [1.0, 1.0, 0.9995, 0.9158, 0.454, 0.0486, 0.0, 0.0, 0.0, 0.0, 0.0],
    'proposed': [1.0, 1.0, 0.9998, 0.9927, 0.9453, 0.7623, 0.3623, 0.0658, 0.0007, 0.0, 0.0]
}

df = pd.DataFrame(data)

plot_comparison(df, "secA_result_cf_sched")