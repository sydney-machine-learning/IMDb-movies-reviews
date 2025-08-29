import pandas as pd
import re
import os

# 表情符号的正则表达式
emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F700-\U0001F77F"  # alchemical symbols
                           u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                           u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                           u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                           u"\U0001FA00-\U0001FA6F"  # Chess symbols, etc.
                           "]+", flags=re.UNICODE)

# 处理单个文件并提取表情符号
def extract_emojis_from_file(file_path):
    df = pd.read_csv(file_path)
    emojis = []

    # 假设你要处理某一列，比如"review"列
    for index, row in df.iterrows():
        text = str(row['Review'])  # 将文本转为字符串，防止非文本类型出错
        found_emojis = emoji_pattern.findall(text)
        emojis.extend(found_emojis)

    return emojis

# 处理多个CSV文件
def process_multiple_files(folder_path):
    all_emojis = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            emojis = extract_emojis_from_file(file_path)
            all_emojis.extend(emojis)

    return all_emojis

# 指定你的CSV文件夹路径
folder_path = '../Reviews'  # 请替换为存储CSV文件的文件夹路径
all_extracted_emojis = process_multiple_files(folder_path)

# 打印所有提取的表情符号
print("Extracted Emojis:", all_extracted_emojis)

# 如果需要保存到文件
with open('extracted_emojis.txt', 'w',encoding='utf-8') as f:
    for emoji_char in all_extracted_emojis:
        f.write(emoji_char + '\n')
