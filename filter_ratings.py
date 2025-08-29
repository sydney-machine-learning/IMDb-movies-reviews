import pandas as pd


df = pd.read_csv('title.ratings.tsv', sep='\t')

# 筛选出averageRating小于x的样本

filtered_df = df[(df['averageRating'] >= 1.0) & (df['averageRating'] < 2.0)]

# 按numVotes降序排序并选择前250个样本
bottom_250_df = filtered_df.sort_values(by='numVotes', ascending=False).head(1000)

# 将结果保存为一个新的TSV文件
bottom_250_df.to_csv('1_star_rated_movies.tsv', sep='\t', index=False)

print("筛选后的前250个样本已保存为 '1_star_rated_movies.tsv' 文件。")