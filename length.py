import pandas as pd
import matplotlib.pyplot as plt

# 存储每个星级的评论长度中位数
median_lengths = []

# 遍历10个CSV文件
for i in range(1, 11):
    # 读取CSV文件
    df = pd.read_csv(f'reviews_rating_{i}.csv')

    # 获取第三列的评论
    reviews = df.iloc[:, 2]

    # 计算评论长度的中位数
    median_length = reviews.str.split().str.len().median()
    median_lengths.append(median_length)

# 绘制柱状图
plt.figure(figsize=(10, 6))

# 调整每条柱子的宽度，比如设为0.8（默认是0.8，也可更小）
plt.bar(range(1, 11), median_lengths, color='skyblue', width=0.65)

# 设置坐标轴标签并调整字体大小
plt.xlabel('Rating', fontsize=28)
plt.ylabel('Median Review Length', fontsize=28)

# 设置 X 轴刻度范围、刻度文本及字体大小
plt.xticks(range(1, 11), fontsize=28)
plt.yticks(fontsize=28)

# 设置 y 轴的取值范围，让最大值显示到1200
plt.ylim(0, 200)

plt.show()



