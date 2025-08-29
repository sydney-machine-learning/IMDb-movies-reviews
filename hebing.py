import os
import pandas as pd
from glob import glob

# 读取当前目录下所有匹配的 CSV 文件
csv_files = glob("imdb_reviews_by_rating*.csv")

# 创建一个空的字典来存储不同 Rating 的数据
rating_data = {}

# 遍历所有 CSV 文件
for file in csv_files:
    df = pd.read_csv(file)  # 读取 CSV 文件
    if "Rating" not in df.columns:
        print(f"Warning: {file} does not contain a 'Rating' column.")
        continue  # 跳过没有 'Rating' 列的文件

    # 遍历所有行，将数据按 Rating 归类
    for _, row in df.iterrows():
        rating = row["Rating"]
        rating_folder = f"Rating_{rating}"

        if rating not in rating_data:
            rating_data[rating] = []

        rating_data[rating].append(row)

# 将分类好的数据存入对应的文件夹
for rating, rows in rating_data.items():
    folder_name = f"Rating_{rating}"
    os.makedirs(folder_name, exist_ok=True)  # 创建文件夹（如果不存在）

    output_file = os.path.join(folder_name, f"reviews_rating_{rating}.csv")

    # 转换为 DataFrame 并写入 CSV
    pd.DataFrame(rows).to_csv(output_file, index=False, encoding="utf-8-sig")

print("CSV 文件已按 Rating 分类存储！")

