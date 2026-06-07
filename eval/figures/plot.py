import matplotlib.pyplot as plt
import numpy as np

# 数据准备
models = ['Text', 'Gen.', 'Cont.']
type_a = np.array([61.4, 54.1, 29.3]) # Hallucination
type_b = np.array([34.6, 42.4, 66.5]) # Boundary Jitter
type_c = np.array([4.0, 3.5, 4.2])    # Semantic

# 设置图表大小 (宽高比更契合双栏或单栏排版)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
width = 0.55

# 画堆叠柱状图 (zorder=3 确保柱子在网格线之上)
p1 = ax.bar(models, type_a, width, label='Type A (Hallucination)', color='#ff9999', edgecolor='black', zorder=3)
p2 = ax.bar(models, type_b, width, bottom=type_a, label='Type B (Boundary Jitter)', color='#66b3ff', edgecolor='black', zorder=3)
p3 = ax.bar(models, type_c, width, bottom=type_a+type_b, label='Type C (Semantic)', color='#99ff99', edgecolor='black', zorder=3)

# 手动精准打标签，解决极小数值(4%)压线重叠的问题
for i in range(len(models)):
    # Type A 标签 (居中)
    ax.text(i, type_a[i]/2, f'{type_a[i]:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold')
    # Type B 标签 (居中)
    ax.text(i, type_a[i] + type_b[i]/2, f'{type_b[i]:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold')
    # Type C 标签 (稍微偏下一点点，并调小字号防止溢出边框)
    ax.text(i, type_a[i] + type_b[i] + type_c[i]/2 - 0.5, f'{type_c[i]:.1f}%', ha='center', va='center', fontsize=11, fontweight='bold')

# Y轴设置：最高给到 105 留出呼吸空间
ax.set_ylim(0, 105)
ax.set_ylabel('Error Distribution (%)', fontsize=12, fontweight='bold')

# 添加虚线网格，增加学术感
ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)

# 图例放置在图表正上方，横向排开 (顶会标配排版)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.12), ncol=3, fontsize=11, frameon=False)

# 去除顶部和右侧的边框线 (Despine)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
# 导出为高精度 PDF
plt.savefig('error_stacked_bar.pdf', format='pdf', dpi=300, bbox_inches='tight')