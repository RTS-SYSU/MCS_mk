import matplotlib.pyplot as plt
import pandas as pd

def plot_comparison(df, filename):
    plt.figure(figsize=(10, 5))

    # Base1
    plt.plot(df["m"], df["base1"],
             marker='o', linestyle='-', linewidth=2, markersize=8,
             label='m<k')

    # Base2
    plt.plot(df["m"], df["base2"],
             marker='s', linestyle='--', linewidth=2, markersize=8,
             label='m=k')

    # Proposed
    plt.plot(df["m"], df["proposed"],
             marker='^', linestyle='-', linewidth=2, markersize=8,color='red',
             label='Proposed')

    # 标题 & 坐标轴
    #plt.title('Performance Comparison vs. m', fontsize=16)
    plt.xlabel('m', fontsize=22)
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
    "m": [1, 2, 3, 4, 5, 6, 7, 8, 9],
    "base1": [0.095510065, 0.143967214, 0.202851464, 0.265953625, 0.32751572, 0.391123088, 0.438083073, 0.470770369, 0.488570093],
    "base2": [0.5155, 0.5275, 0.5229, 0.528, 0.525, 0.5302, 0.5241, 0.5306, 0.5269],
    "proposed": [0.935735074, 0.834073417, 0.815706675, 0.810978202, 0.806780055, 0.802102554, 0.791467003, 0.7694576, 0.750631757]
}

df = pd.DataFrame(data)

plot_comparison(df, "secA_result_mk_utility")