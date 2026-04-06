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
    plt.ylabel('Schedulability Ratio', fontsize=22)

    # 范围控制（和你之前风格一致）
    plt.ylim(0, 1.05)

    # 网格 & 图例
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(f"{filename}.pdf", dpi=300)
    print(f"Plot saved to {filename}.pdf")
    # plt.show()

data = {
    "m": [1,2,3,4,5,6,7,8,9],
    "base1": [0.9321,0.7035,0.6631,0.6534,0.6455,0.6443,0.6207,0.5851,0.5414],
    "base2": [0.5155,0.5275,0.5229,0.528,0.525,0.5302,0.5241,0.5306,0.5269],
    "proposed": [0.9923,0.8711,0.8372,0.825,0.8167,0.8088,0.7961,0.7726,0.7519]
}

df = pd.DataFrame(data)

plot_comparison(df, "secA_result_mk_sched")