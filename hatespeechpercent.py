import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 存放每个文件中标签0（hateful）和标签1（offensive）的百分比
label0_percent = []  # hateful
label1_percent = []  # offensive

# 遍历 classified_file1.csv 到 classified_file10.csv 文件
for i in range(1, 11):
    file_path = f"classified_file{i}.csv"
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # 计算 predicted_class 列中各标签的比例（转换为百分比）
        counts = df['predicted_class'].value_counts(normalize=True) * 100
        # 只记录标签0和标签1的比例，若不存在则默认为0%
        label0_percent.append(counts.get(0, 0))
        label1_percent.append(counts.get(1, 0))
    else:
        print(f"文件 {file_path} 不存在。")
        label0_percent.append(0)
        label1_percent.append(0)

# x 轴刻度位置，代表 10 个评论星级（1-10）
x = np.arange(10)
bar_width = 0.4  # 每个柱子的宽度

plt.figure(figsize=(12, 8))

# 绘制标签0和标签1的柱状图
plt.bar(x - bar_width/2, label0_percent, width=bar_width, label='Hateful Speech', color='skyblue')
plt.bar(x + bar_width/2, label1_percent, width=bar_width, label='Offensive Language', color='orange')

# 设置坐标轴标签和标题
plt.xlabel('Rating', fontsize=28)
plt.ylabel('Percentage', fontsize=28)

# 设置 x 轴刻度为 1 到 10
plt.xticks(x, [str(i) for i in range(1, 11)], fontsize=28)
plt.yticks(fontsize=28)

# 动态设置 y 轴最大值为所有数据中的最大值的 1.1 倍，使其略高于最大 bar 值
max_value = max(max(label0_percent), max(label1_percent))
plt.ylim(0, max_value * 1.1)
plt.legend(fontsize=28)
plt.show()
