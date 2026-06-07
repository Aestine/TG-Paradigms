import matplotlib.pyplot as plt
import numpy as np

# 设置学术论文极简风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# ================= 调大全局字体 =================
plt.rcParams['axes.labelsize'] = 13     # 轴标签字体调大
plt.rcParams['xtick.labelsize'] = 12    # X轴刻度字体调大
plt.rcParams['ytick.labelsize'] = 12    # Y轴刻度字体调大
plt.rcParams['legend.fontsize'] = 11    # 图例字体调大
plt.rcParams['axes.titlesize'] = 14     # 子图标题字体调大

# ================= 数据准备 =================
backbones = [0.5, 1.5, 2.2, 4.0, 8.0]
# 根据最新 Table 同步了 8B 的数据 (Cont 54.5, Gen 35.1)
miou_text = [16.2, 19.9, 20.8, 23.9, 27.5]
miou_cont = [43.4, 46.6, 46.6, 56.3, 57.1]
miou_gen  = [21.9, 28.9, 28.1, 32.3, 35.1]

# Latency 数据保持不变
latency_text = [480, 540, 663, 1250, 2150]
latency_cont = [590, 660, 794, 1480, 2540]
latency_gen  = [920, 1050, 1379, 2560, 4380]

# ================= 开始绘图 =================
# 稍微放大画布以容纳更大的字体 (6.5, 3.0 -> 7.5, 3.5)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
colors = ['#d62728', '#1f77b4', '#ff7f0e'] # 红(Text), 蓝(Cont), 橙(Gen)
markers = ['s', 'o', '^']

# ------ 子图 (a): Scaling Curves ------
ax1.plot(backbones, miou_text, marker=markers[0], color=colors[0], linewidth=2.5, markersize=7, label='Text')
ax1.plot(backbones, miou_cont, marker=markers[1], color=colors[1], linewidth=2.5, markersize=7, label='Cont.')
ax1.plot(backbones, miou_gen,  marker=markers[2], color=colors[2], linewidth=2.5, markersize=7, label='Gen.')

ax1.set_xlabel('Backbone Size (B)')
ax1.set_ylabel('mIoU (Charades-STA)')
ax1.set_title('(a) Scaling Curve', pad=10)
ax1.set_xticks([0.5, 2.2, 4.0, 8.0])

# 图例放置在留白处
ax1.legend(loc='upper right', bbox_to_anchor=(0.98, 0.85), frameon=True, fancybox=False, edgecolor='black', handlelength=1.5)

# ------ 子图 (b): Latency-Accuracy Pareto ------
sizes = [50, 80, 120, 170, 230] # 散点也稍微调大一点
for i in range(len(backbones)):
    ax2.scatter(latency_text[i], miou_text[i], s=sizes[i], c=colors[0], marker=markers[0], alpha=0.8, edgecolors='white', linewidth=0.5)
    ax2.scatter(latency_cont[i], miou_cont[i], s=sizes[i], c=colors[1], marker=markers[1], alpha=0.8, edgecolors='white', linewidth=0.5)
    ax2.scatter(latency_gen[i],  miou_gen[i],  s=sizes[i], c=colors[2], marker=markers[2], alpha=0.8, edgecolors='white', linewidth=0.5)

# 真实 Pareto 前沿计算
pareto_x = [latency_text[0], latency_text[1], latency_cont[0], latency_cont[1], latency_cont[3], latency_cont[4]]
pareto_y = [miou_text[0], miou_text[1], miou_cont[0], miou_cont[1], miou_cont[3], miou_cont[4]]
ax2.plot(pareto_x, pareto_y, linestyle='--', color='gray', linewidth=1.5, zorder=0, label='Pareto Frontier')

ax2.set_xlabel('Inference Latency (ms)')
ax2.set_title('(b) Pareto Frontier', pad=10)
ax2.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black', handlelength=1.5)

# 调整布局
plt.tight_layout(pad=0.5, w_pad=2.0)
plt.savefig('scaling_pareto.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("scaling_pareto.pdf (大字体更新版) 已生成！")