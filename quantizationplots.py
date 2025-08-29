import pandas as pd
import matplotlib.pyplot as plt
import ast

# === 读取CSV ===
df = pd.read_csv('imdb_reviews_tt11315808_Joker2.csv')
df = df[df['Rating'].apply(lambda x: str(x).isdigit())]
df['Rating'] = df['Rating'].astype(int)

# === 解析 Target 为字典 ===
df['Target_dict'] = df['Target'].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})

# 标签列表
aspects = ['Storyline', 'Acting', 'Cinematography', 'Soundtrack', 'Rewatchability']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
aspect_color_map = dict(zip(aspects, colors))

# 初始化结果
results = []
review_counts = {}

# 遍历每个星级
for rating, group in df.groupby('Rating'):
    review_counts[rating] = len(group)
    score_sums = dict.fromkeys(aspects, 0)

    for tag_dict in group['Target_dict']:
        for tag in aspects:
            score_sums[tag] += tag_dict.get(tag, 0.0)

    row = {'Rating': rating}
    for tag in aspects:
        row[tag] = score_sums[tag] / review_counts[rating] if review_counts[rating] > 0 else 0
    results.append(row)

# 创建 DataFrame
result_df = pd.DataFrame(results).sort_values('Rating').set_index('Rating')
review_count_series = pd.Series(review_counts).sort_index()

# === 绘图 ===
fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
bars = result_df.plot(kind='bar', stacked=True, ax=ax, color=colors)

# 设置字体大小
plt.xlabel('Rating', fontsize=14)
plt.ylabel('Average Probability', fontsize=14)
plt.xticks(rotation=0)
ax.tick_params(axis='x', labelsize=14)
ax.tick_params(axis='y', labelsize=14)

# 添加评论数量标签
for i, (rating, total_prob) in enumerate(result_df.sum(axis=1).items()):
    count = review_count_series[rating]
    ax.text(i, total_prob + 0.02, f'n={count}', ha='center', va='bottom', fontsize=14)

# === 提高 y 轴上限，留出空间 ===
y_max = result_df.sum(axis=1).max()
ax.set_ylim(0, y_max * 1.25)

# === 图例样式 ===
legend = ax.legend(
    labels=aspects,
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    ncol=3,
    fontsize=14,
    handletextpad=0.2,
    columnspacing=0.8,
    labelspacing=0.4,
    borderaxespad=0.2
)
legend.get_frame().set_facecolor('white')
legend.set_title(None)

# === 结束图像设置 ===
plt.tight_layout()
plt.savefig("Joker2.png", dpi=300, bbox_inches='tight')
plt.show()

