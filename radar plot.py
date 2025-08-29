"""import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import ast

# 定义10种不同的颜色
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEEAD',
          '#D4A5A5', '#9B59B6', '#3498DB', '#E74C3C', '#2ECC71']


# 处理所有10个文件
def process_emotion_data():
    all_emotions_percentages = {}

    for rating in range(1, 11):
        filename = f'predicted_reviews_rating_{rating}.csv'
        df = pd.read_csv(filename)

        emotion_counts = {}
        total_emotions = 0

        for emotions_str in df['Predicted_Emotions']:
            emotions_list = ast.literal_eval(emotions_str)

            for emotion_entry in emotions_list:
                emotion = emotion_entry.split('(')[0]
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                total_emotions += 1

        emotion_percentages = {
            emotion: (count / total_emotions) * 100
            for emotion, count in emotion_counts.items()
        }

        sorted_emotions = sorted(emotion_percentages.items(),
                                 key=lambda x: x[1],
                                 reverse=True)[:5]

        all_emotions_percentages[rating] = sorted_emotions

    return all_emotions_percentages


# 创建单个雷达图
def create_radar_chart(rating, emotions_data, color,
                       label_font_size=12, tick_font_size=10, outline_width=2):
    emotions = emotions_data[rating]
    categories = [emotion[0] for emotion in emotions]
    values = [emotion[1] for emotion in emotions]

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    # 创建新图形
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)

    # 设置角度和方向
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # 设置网格线和范围
    max_value = max(values)  # 不再乘以1.2，确保不超过边界
    max_int_value = int(np.ceil(max_value))  # 取上限整数
    ax.set_ylim(0, max_int_value)

    # 设置整数刻度
    step = max(1, max_int_value // 5)  # 动态步长，至少为1
    tick_values = list(range(0, max_int_value + step, step))
    if tick_values[-1] > max_int_value:  # 确保不超过最大值
        tick_values = tick_values[:-1]
    ax.set_yticks(tick_values)

    # 设置标签并控制字体大小
    plt.xticks(angles[:-1], categories, size=label_font_size)

    # 设置刻度标签字体大小
    ax.tick_params(axis='y', labelsize=tick_font_size)

    # 调整标签位置避免重叠
    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle in (0, pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        label.set_position((1.1, 0))

    # 绘制数据，控制外围线粗细
    ax.plot(angles, values, linewidth=outline_width, linestyle='solid', color=color)
    ax.fill(angles, values, color, alpha=0.25)

    # 显示图形
    plt.tight_layout()
    plt.show()


# 主程序
def main(label_font_size=12, tick_font_size=10, outline_width=2):
    # 处理数据
    emotion_data = process_emotion_data()

    # 打印每个rating的前5个情感及其百分比
    for rating in range(1, 11):
        print(f"\nRating {rating} Top 5 Emotions:")
        for emotion, percentage in emotion_data[rating]:
            print(f"{emotion}: {percentage:.2f}%")

    # 为每个rating创建单独的雷达图
    for rating in range(1, 11):
        create_radar_chart(rating, emotion_data, COLORS[rating - 1],
                           label_font_size, tick_font_size, outline_width)


if __name__ == "__main__":
    # 可以在这里调整字体大小和外围线粗细
    main(label_font_size=28, tick_font_size=28, outline_width=4)"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from math import pi
import ast

# -------------------------
# 1. 数据读取与情感计数
#    Data reading and emotion counting
# -------------------------
def read_and_count_emotions(file_indices):
    """
    中文功能说明:
      给定文件编号列表, 读取对应的CSV文件, 将所有行的Predicted_Emotions进行统计.
      最终返回一个字典, 形如 {emotion: count, ...} 以及情感出现的总次数 total_count.

    English Explanation:
      Given a list of file indices, read the corresponding CSV files.
      Parse the 'Predicted_Emotions' column, and accumulate counts in a dictionary
      of the form {emotion: count, ...}, and also return total_count (the sum of all counts).
    """
    emotion_counts = {}
    total_count = 0

    for idx in file_indices:
        filename = f"predicted_reviews_rating_{idx}.csv"
        df = pd.read_csv(filename)

        for emotions_str in df["Predicted_Emotions"]:
            emotions_list = ast.literal_eval(emotions_str)
            for emotion_item in emotions_list:
                emotion_name = emotion_item.split("(")[0]
                emotion_counts[emotion_name] = emotion_counts.get(emotion_name, 0) + 1
                total_count += 1

    return emotion_counts, total_count


def get_top_5_emotions(emotion_counts, total_count):
    """
    中文功能说明:
      给定情感计数字典和总次数, 计算各情感出现的百分比并排序, 返回占比最高的前5个情感及其百分比.

    English Explanation:
      Given the emotion count dictionary and total count, compute the percentage for each emotion,
      sort them in descending order, and return the top 5 emotions and their percentages.
    """
    emotion_percentages = {
        emotion: (count / total_count) * 100
        for emotion, count in emotion_counts.items()
    }
    sorted_emotions = sorted(emotion_percentages.items(), key=lambda x: x[1], reverse=True)
    return sorted_emotions[:5]

# -------------------------
# 2. 雷达图绘制
#    Radar chart plotting
# -------------------------
def create_radar_chart(group_name, top_emotions, color, line_width=3):
    """
    中文功能说明:
      给定组名(如 "Group 1-3")、该组前5情感(含情感名称和百分比)、线条颜色和线条宽度,
      生成并展示一个雷达图, 去掉图表标题, 并去除右上角的图例.

    English Explanation:
      Given a group name (e.g., "Group 1-3"), the top 5 emotions (names and percentages),
      a color for the line, and line width, generate and display a radar chart with no title
      and no legend in the top-right corner.
    """
    categories = [item[0] for item in top_emotions]
    values = [item[1] for item in top_emotions]

    N = len(categories)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    values += values[:1]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, polar=True)
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    max_value = max(values)
    max_int_value = int(np.ceil(max_value))
    ax.set_ylim(0, max_int_value)

    step = max(1, max_int_value // 5)
    tick_values = list(range(0, max_int_value + step, step))
    if tick_values[-1] > max_int_value:
        tick_values = tick_values[:-1]
    ax.set_yticks(tick_values)

    plt.xticks(angles[:-1], categories, size=12)

    for label, angle in zip(ax.get_xticklabels(), angles[:-1]):
        if angle in (0, pi):
            label.set_horizontalalignment('center')
        elif 0 < angle < pi:
            label.set_horizontalalignment('left')
        else:
            label.set_horizontalalignment('right')
        label.set_position((1.1, 0))

    # 绘制雷达图 (加粗线条, 去掉标题, 去掉图例)
    ax.plot(angles, values, linewidth=line_width, linestyle='solid', color=color)
    ax.fill(angles, values, color, alpha=0.25)

    # 设置情感标签（类别）的字体大小为16
    plt.xticks(angles[:-1], categories, size=35)

    # 设置雷达图中数字（径向刻度）的字体大小为16
    ax.tick_params(axis='y', labelsize=35)

    # 去掉标题 (Remove the title)
    # plt.title(group_name, size=16, y=1.08)

    # 去掉图例 (Remove the legend)
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.1))

    plt.tight_layout()
    plt.show()

# -------------------------
# 3. 主流程
#    Main workflow
# -------------------------
def main():
    """
    中文功能说明:
      - 定义3个分组: 第1~3个文件, 第4~6个文件, 第7~10个文件
      - 对每个分组, 读取并统计所有情感出现次数及占比
      - 选取占比最高的前5个情感
      - 绘制并展示雷达图(使用黄色/橙色/蓝色, 加粗线条, 去掉标题, 去掉右上角图例)

    English Explanation:
      - Define three groups of file indices: (1–3), (4–6), (7–10).
      - For each group, read and parse the CSV files, accumulate emotion counts and percentages.
      - Select the top 5 emotions by percentage.
      - Plot and display a radar chart (using yellow/orange/blue colors, thicker lines,
        no title, and no legend in the top-right corner).
    """
    groups = {
        "Group 1-3": [1, 2, 3],
        "Group 4-6": [4, 5, 6],
        "Group 7-10": [7, 8, 9, 10]
    }

    group_color_map = {
        "Group 1-3": "green",
        "Group 4-6": "orange",
        "Group 7-10": "skyblue"
    }

    for group_name, file_list in groups.items():
        emotion_counts, total_count = read_and_count_emotions(file_list)
        top_5_emotions = get_top_5_emotions(emotion_counts, total_count)

        print(f"\n{group_name} Top 5 Emotions:")
        for emotion, pct in top_5_emotions:
            print(f"{emotion}: {pct:.2f}%")

        create_radar_chart(
            group_name,
            top_5_emotions,
            color=group_color_map[group_name],
            line_width=3
        )


if __name__ == "__main__":
    main()
