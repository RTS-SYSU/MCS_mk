import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 1. 从xlsx文件读取数据
file_path = 'schedulability_comparison_4core.xlsx'  # ⚠️ 请替换为你的实际文件路径
df = pd.read_excel(file_path)

# 清理列名（防止Excel表头带有隐藏空格导致KeyError）
df.columns = df.columns.str.strip()

# 2. 设置画布
plt.figure(figsize=(8, 6))

# 3. 绘制折线 (保留你提供的代码)
plt.plot(df["Utilization"], df["Our_Ratio"],
         color='#d62728', marker='o', linestyle='-', linewidth=2, markersize=8,
         label='Our Proposal')

plt.plot(df["Utilization"], df["AMC_Ratio"],
         color='#1f77b4', marker='s', linestyle='--', linewidth=2, markersize=8,
         label='AMC-rtb-WH')

# 4. 设置坐标轴标签和刻度
plt.xlabel(r'$U_{avg}^{LO}$', fontsize=18)
plt.xticks(np.arange(0.4, 1.0, 0.05))  # X轴：0.4~0.9，间隔0.1
plt.xlim(0.4, 0.8)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.ylabel('Schedulability Ratio', fontsize=18)
plt.yticks(np.arange(0, 1.1, 0.1))    # Y轴：0.0~1.0，间隔0.1
plt.ylim(-0.01, 1.01)                        # 限制Y轴显示范围
plt.grid(True, linestyle='--', alpha=0.6)
# 5. 添加图例
plt.legend(loc='best', fontsize=14)

# 6. 调整布局并保存为PDF
plt.tight_layout()
output_file = 'schedulability_plot.pdf'
plt.savefig(output_file, format='pdf', dpi=450, bbox_inches='tight')
print(f"✅ 图片已成功保存为: {output_file}")

# 显示图片（可选）
plt.show()