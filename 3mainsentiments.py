

import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import defaultdict
import numpy as np

# 读取CSV文件
# Read CSV file
df = pd.read_csv('imdb_reviews_tt7286456_Joker.csv')

# 定义所有可能的情绪和指定颜色
# Define all possible sentiments and assigned colors
sentiment_to_color = {
    "Admiration": '#d62728',  # Blue
    "Amusement": '#2ca02c',    # Green
    "Anger": '#d62728',        # Red
    "Annoyance": '#8a2be2',    # Gray
    "Approval": '#98df8a',     # Light Green
    "Caring": '#9467bd',       # Purple
    "Confusion": '#006400',    # Dark Green
    "Curiosity": '#7f7f7f',    # Orange
    "Desire": '#7f7f7f',       # Deep Purple
    "Disappointment": '#ff7f0e',  # Orange
    "Disapproval": '#17becf',  # Light Blue
    "Disgust": '#1f77b4',      # Dark Blue
    "Embarrassment": '#f2a4b6',  # Pink
    "Excitement": '#ffdd47',   # Yellow
    "Fear": '#8b0000',         # Dark Red
    "Gratitude": '#ffdd47',    # Yellow
    "Grief": '#f2a4b6',        # Pink
    "Joy": '#2ca02c',          # Green
    "Love": '#1f77b4',         # Red
    "Nervousness": '#7f7f7f',   # Gray
    "Optimism": '#9467bd',     # Purple
    "Pride": '#006400',        # Dark Green
    "Realization": '#ff7f0e',  # Orange
    "Relief": '#98df8a',       # Light Green
    "Remorse": '#8a2be2',      # Purple
    "Sadness": '#aec7e8',      # Light Blue
    "Surprise": '#ffdd47',     # Yellow
    "Neutral": '#ffdd47'       # Yellow
}


# 累积每个rating的情绪概率
# Accumulate sentiment probabilities for each rating
rating_sentiment_scores = defaultdict(lambda: defaultdict(float))
rating_counts = defaultdict(int)

# 遍历每一行，解析Sentiment列
# Parse Sentiment column
for idx, row in df.iterrows():
    try:
        rating = int(row['Rating'])
    except ValueError:
        continue  # 跳过没有rating的评论
    sentiments = row['Sentiment']
    matches = re.findall(r'(\w+)\(([\d.]+)\)', sentiments)

    for sentiment, prob in matches:
        if sentiment in sentiment_to_color:
            rating_sentiment_scores[rating][sentiment] += float(prob)

    rating_counts[rating] += 1

# 平均化
# Average the sentiment scores
for rating in rating_sentiment_scores:
    for sentiment in rating_sentiment_scores[rating]:
        rating_sentiment_scores[rating][sentiment] /= rating_counts[rating]

# 准备画图数据
# Prepare data for plotting
ratings = list(range(1, 11))
top3_sentiments_per_rating = {}
legend_sentiments = set()  # 用于记录出现在图中的情绪
for rating in ratings:
    if rating in rating_sentiment_scores:
        sorted_sentiments = sorted(rating_sentiment_scores[rating].items(), key=lambda x: x[1], reverse=True)
        top3_sentiments_per_rating[rating] = sorted_sentiments[:3]
        legend_sentiments.update([sentiment for sentiment, _ in top3_sentiments_per_rating[rating]])
    else:
        top3_sentiments_per_rating[rating] = []

# 画图
# Plotting
fig, ax = plt.subplots(figsize=(18, 9))
bar_width = 0.2

for i, rating in enumerate(ratings):
    sentiments = top3_sentiments_per_rating[rating]
    for j, (sentiment, avg_prob) in enumerate(sentiments):
        x_base = i
        offset = (j - 1) * bar_width
        ax.bar(x_base + offset, avg_prob, width=bar_width,
               label=sentiment if (i == 0 and sentiment in legend_sentiments) else "",
               color=sentiment_to_color[sentiment],
               edgecolor='black', linewidth=0.7)

# 设置x轴
# Set x-axis
ax.set_xticks(range(len(ratings)))
ax.set_xticklabels(ratings)
ax.set_xlabel('Rating', fontsize=18)
ax.set_ylabel('Sentiment Probability', fontsize=18)

# 设置x和y轴刻度字体大小
# Set the font size for x and y axis ticks
ax.tick_params(axis='both', which='major', labelsize=18)

# 只为出现在图中的情绪创建legend元素
# Create legend only for sentiments present in the plot
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=sentiment_to_color[sentiment], edgecolor='black', label=sentiment)
                   for sentiment in legend_sentiments]

legend = ax.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(0.02, 0.98),
    ncol=3,
    fontsize=18,
    handletextpad=0.2,
    columnspacing=0.8,
    labelspacing=0.4,
    borderaxespad=0.2
)

legend.get_frame().set_facecolor('white')  # 设置背景为白色
legend.set_title(None)  # 不显示标题

# 保存图片
plt.savefig('D:/Pytorch Training/4cases/jokerbarchart.png', dpi=300, bbox_inches='tight')

# 显示图
plt.show()
