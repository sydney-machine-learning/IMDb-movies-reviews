import pandas as pd
import glob

# 获取所有 reviews_rating_1.csv 到 reviews_rating_10.csv 文件路径
file_paths = glob.glob("reviews_rating_*.csv")  # 确保文件在当前目录下

# 创建一个空的集合来存储唯一的 MovieID
unique_movie_ids = set()

# 逐个读取文件并提取 MovieID
for file_path in file_paths:
    df = pd.read_csv(file_path)  # 读取CSV文件
    if df.shape[1] > 0:  # 确保文件不为空
        unique_movie_ids.update(df.iloc[:, 0].unique())  # 提取第一列的唯一值并加入集合

# 计算唯一 MovieID 的总数
print(f"Total unique MovieID count: {len(unique_movie_ids)}")
