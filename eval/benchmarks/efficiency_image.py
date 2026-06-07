import matplotlib.pyplot as plt
import numpy as np

# 设置全局字体大小
plt.rcParams.update({'font.size': 11})

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
width = 0.35  # 柱子宽度
paradigms = ['Text', 'Dist.', 'Gen.']
x = np.arange(len(paradigms))

# ========== 这里填入你的真实数据 ==========
# 真实数据 (a) SmolVLM-2.2B
throughput_smol = [2.81, 4.30, 4.50]
latency_smol = [663, 794, 1379]

# 占位数据 (b) FastVLM-1.5B (跑出数据后替换这里)
throughput_fast = [0, 0, 0]
latency_fast = [0, 0, 0]

# 占位数据 (c) Molmo2-4B (跑出数据后替换这里)
throughput_molmo = [0, 0, 0]
latency_molmo = [0, 0, 0]
# ==========================================

data = [
    (throughput_smol, latency_smol, '(a) SmolVLM-2.2B'),
    (throughput_fast, latency_fast, '(b) FastVLM-1.5B'),
    (throughput_molmo, latency_molmo, '(c) Molmo2-4B')
]

for i, (thru, lat, title) in enumerate(data):
    ax1 = axes[i]
    ax2 = ax1.twinx()

    # 绘制双柱状图 (加上 zorder=3 让柱子显示在网格线前方)
    b1 = ax1.bar(x - width / 2, thru, width, label='Throughput', color='#4C72B0', edgecolor='black', zorder=3)
    b2 = ax2.bar(x + width / 2, lat, width, label='Latency', color='#C44E52', edgecolor='black', zorder=3)

    # 设置X轴
    ax1.set_xticks(x)
    ax1.set_xticklabels(paradigms, fontsize=12)
    ax1.set_title(title, fontsize=14, pad=10)

    # 设置Y轴标签（仅在最左边和最右边显示，保持画面干净）
    if i == 0:
        ax1.set_ylabel('Throughput (samples/s)', color='#4C72B0', fontsize=12, fontweight='bold')
    if i == 2:
        ax2.set_ylabel('Latency (ms/query)', color='#C44E52', fontsize=12, fontweight='bold')

    ax1.tick_params(axis='y', labelcolor='#4C72B0')
    ax2.tick_params(axis='y', labelcolor='#C44E52')

    # Y轴的范围留出一些头部空间 (乘以1.25)，避免柱子上方的数值被遮挡
    ax1.set_ylim(0, max(thru) * 1.25)
    ax2.set_ylim(0, max(lat) * 1.25)

    # 添加数值标签
    for rect in b1:
        height = rect.get_height()
        ax1.annotate(f'{height:.2f}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),  # 垂直偏移3个像素
                     textcoords="offset points",
                     ha='center', va='bottom', color='#4C72B0', fontsize=10, fontweight='bold')
    for rect in b2:
        height = rect.get_height()
        ax2.annotate(f'{height}',
                     xy=(rect.get_x() + rect.get_width() / 2, height),
                     xytext=(0, 3),
                     textcoords="offset points",
                     ha='center', va='bottom', color='#C44E52', fontsize=10, fontweight='bold')

    # 添加横向网格线 (zorder=0 确保网格线在柱子后方)
    ax1.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

# 添加全局图例 (放在三张图的上方中间)
fig.legend([b1, b2], ['Throughput (samp/s) - Left Axis', 'Latency (ms/query) - Right Axis'],
           loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2, frameon=False, fontsize=13)

plt.tight_layout()

# 导出为PNG供预览，导出为PDF供LaTeX插入 (绝对不会失真)
plt.savefig('efficiency_bars.png', bbox_inches='tight', dpi=300)
plt.savefig('efficiency_bars.pdf', bbox_inches='tight')
plt.show()