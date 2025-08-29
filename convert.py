import emoji

# 读取文件内容并转换表情符号为文本描述
def convert_emojis_in_file(input_file, output_file):
    # 读取txt文件
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # 将表情符号转换为文本描述
    converted_content = emoji.demojize(content)

    # 写入转换后的内容到新的txt文件
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(converted_content)

# 示例文件路径
input_file = 'extracted_emojis.txt'  # 原始包含表情符号的文件
output_file = 'convert_to_words .txt'  # 转换为文本描述后的文件

# 执行转换
convert_emojis_in_file(input_file, output_file)

print("表情符号已转换为文本描述并保存到新文件。")
