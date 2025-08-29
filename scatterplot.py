import pandas as pd
import ast
import numpy as np
import matplotlib.pyplot as plt
import math

# --------------------- 数据处理 / Data Processing ---------------------
file_paths = [f"processed_classified_file{i}.csv" for i in range(1, 11)]
target_labels = ["Storyline", "Acting", "Cinematography", "Soundtrack", "Rewatchability"]

# 存储每个标签的比例（Y轴）和平均强度（用于散点大小）
label_ratios = {label: [] for label in target_labels}
label_avg_intensity = {label: [] for label in target_labels}

for file_path in file_paths:
    df = pd.read_csv(file_path)
    count_dict = {label: 0 for label in target_labels}
    sum_dict = {label: 0.0 for label in target_labels}
    total_count = 0

    for target_str in df["Target"].dropna():
        try:
            target_dict = ast.literal_eval(target_str)
            for label in target_labels:
                if label in target_dict:
                    count_dict[label] += 1
                    sum_dict[label] += target_dict[label]
                    total_count += 1
        except (SyntaxError, ValueError):
            continue

    for label in target_labels:
        ratio = (count_dict[label] / total_count * 100) if total_count > 0 else 0
        avg_intensity = (sum_dict[label] / count_dict[label]) if count_dict[label] > 0 else 0
        label_ratios[label].append(ratio)
        label_avg_intensity[label].append(avg_intensity)

# --------------------- 绘图 / Plotting ---------------------
plt.style.use('default')
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

x_values = list(range(1, 11))
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for label, color in zip(target_labels, colors):
    y_vals = label_ratios[label]
    intensities = label_avg_intensity[label]
    if len(intensities) > 0:
        sizes = np.interp(intensities, (min(intensities), max(intensities)), (50, 350)).tolist()
    else:
        sizes = [50 for _ in intensities]
    ax.scatter(
        x_values,
        y_vals,
        s=sizes,
        c=color,
        marker='.',
        alpha=0.7,
        label=label
    )
    ax.plot(x_values, y_vals, color=color, lw=1.5, alpha=0.7)

ax.set_xlabel("Rating", fontsize=10)
ax.set_ylabel("Percentage (%)", fontsize=10)
ax.set_xticks(x_values)
ax.tick_params(axis='x', labelsize=10)
ax.tick_params(axis='y', labelsize=10)
ax.grid(True, color='lightgray', linestyle='--', alpha=0.7)

# 动态计算 y 轴下界和上界，使其为 5 的倍数
all_ratios = [r for label in label_ratios for r in label_ratios[label]]
min_y_val = min(all_ratios) if all_ratios else 0
max_y_val = max(all_ratios) if all_ratios else 35
y_bottom = math.floor(min_y_val / 5) * 5
y_top = math.ceil(max_y_val / 5) * 5
ax.set_ylim(y_bottom, y_top)
ax.set_yticks(np.arange(y_bottom, y_top + 1, 5))

# 将图例放置在图内部左上角，且紧凑排列
legend = ax.legend(
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    ncol=3,
    fontsize=10,
    handletextpad=0.2,
    columnspacing=0.8,
    labelspacing=0.4,
    borderaxespad=0.2
)
legend.get_frame().set_facecolor('white')

plt.savefig("2d_label_scatter_dots_lines.png", dpi=300, bbox_inches='tight')
plt.show()
