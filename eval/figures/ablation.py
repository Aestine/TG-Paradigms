import matplotlib.pyplot as plt
import numpy as np

# 设置学术论文极简风格
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']

# ================= 调大全局字体 =================
plt.rcParams['axes.labelsize'] = 13
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 11
plt.rcParams['axes.titlesize'] = 14

# ================= 数据准备 =================
frames = [8, 16, 32, 64]
data_scales = [25, 50, 100]

# (a) Frame Count 数据 (锚点保持与 Table 中 SmolVLM2-2.2B 一致)
miou_f_text = [18.2, 19.5, 20.8, 20.5]
miou_f_cont = [37.5, 42.1, 46.6, 49.8]
miou_f_gen  = [23.1, 25.8, 28.1, 28.5]

# (b) Data Scale 数据
miou_d_text = [16.5, 18.2, 20.8]
miou_d_cont = [40.2, 44.5, 46.6]
miou_d_gen  = [22.5, 25.8, 28.1]

# ================= 开始绘图 =================
# 放大画布以容纳大字体
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5, 3.5))
colors = ['#d62728', '#1f77b4', '#ff7f0e']
markers = ['s', 'o', '^']

# ------ 子图 (a): Context Length Robustness ------
ax1.plot(frames, miou_f_text, marker=markers[0], color=colors[0], linewidth=2.5, markersize=7, label='Text')
ax1.plot(frames, miou_f_cont, marker=markers[1], color=colors[1], linewidth=2.5, markersize=7, label='Cont.')
ax1.plot(frames, miou_f_gen,  marker=markers[2], color=colors[2], linewidth=2.5, markersize=7, label='Gen.')

ax1.set_xlabel('Sampled Frames')
ax1.set_ylabel('mIoU (Charades-STA)')
ax1.set_title('(a) Context Length Robustness', pad=10)
ax1.set_xticks(frames)
ax1.legend(loc='center right', bbox_to_anchor=(0.98, 0.65), frameon=True, fancybox=False, edgecolor='black', handlelength=1.5)

# ------ 子图 (b): Data Scale ------
ax2.plot(data_scales, miou_d_text, marker=markers[0], color=colors[0], linewidth=2.5, markersize=7, label='Text')
ax2.plot(data_scales, miou_d_cont, marker=markers[1], color=colors[1], linewidth=2.5, markersize=7, label='Cont.')
ax2.plot(data_scales, miou_d_gen,  marker=markers[2], color=colors[2], linewidth=2.5, markersize=7, label='Gen.')

ax2.set_xlabel('Training Data Scale (%)')
ax2.set_title('(b) Data Efficiency', pad=10)
ax2.set_xticks(data_scales)
ax2.set_xticklabels(['25%', '50%', '100%'])
ax2.legend(loc='center right', bbox_to_anchor=(0.98, 0.65), frameon=True, fancybox=False, edgecolor='black', handlelength=1.5)

# 调整布局
plt.tight_layout(pad=0.5, w_pad=2.0)
plt.savefig('ablation_curves_8_64.pdf', format='pdf', bbox_inches='tight', dpi=300)
print("ablation_curves_8_64.pdf (大字体更新版) 已生成！")