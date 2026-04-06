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
    plt.ylabel('Normalized Utility', fontsize=22)

    # 范围控制（和你之前风格一致）
    plt.ylim(0, 1.01)

    # 网格 & 图例
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", dpi=300)
    print(f"Plot saved to {filename}.pdf")
    # plt.show()

data = {
    "cf": [1, 1.2, 1.4, 1.6, 1.8, 2, 2.2, 2.4, 2.6, 2.8, 3],
    "base1": [0.504730921, 0.504036666, 0.502518229, 0.493201558, 0.431203096, 0.264475661, 0.056767993, 0.001461857, 0.0, 0.0, 0.0],
    "base2": [1.0, 1.0, 0.9995, 0.9158, 0.454, 0.0486, 0.0, 0.0, 0.0, 0.0, 0.0],
    "proposed": [1.0, 1.0, 0.999727996, 0.99047367, 0.930087518, 0.722824723, 0.314732514, 0.04988889, 0.00041328, 0.0, 0.0]
}

df = pd.DataFrame(data)

plot_comparison(df, "secA_result_cf_utility")