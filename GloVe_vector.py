import numpy as np
import pandas as pd
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# 加载 GloVe 向量的函数
def load_glove_vectors(glove_file):
    glove_vectors = {}
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]  # 获取词汇
            vector = np.asarray(values[1:], dtype='float32')  # 获取对应的向量
            glove_vectors[word] = vector
    return glove_vectors

# 读取CSV文件中的词组
def load_phrases_from_csv(file_path):
    df = pd.read_csv(file_path)
    phrases = df['phrase'].tolist()  # 假设词组在名为'phrases_column_name'的列中
    return phrases

# 示例读取文件
#phrases = load_phrases_from_csv('../high_frequency_words/1_star_high_frequency.csv')
# 3. 将词组转换为向量（每个词组的单词取平均）
def phrase_to_vector(phrase, glove_vectors, vector_size=50):
    words = phrase.split()  # 按空格分词
    word_vectors = [glove_vectors[word] for word in words if word in glove_vectors]

    if len(word_vectors) == 0:
        return np.zeros(vector_size)  # 如果没有找到词，返回全零向量

    return np.mean(word_vectors, axis=0)


def file_to_vector(phrases, glove_vectors, vector_size=50):
    phrase_vectors = [phrase_to_vector(phrase, glove_vectors, vector_size) for phrase in phrases]

    # 取词组向量的平均，表示整个文件的向量
    if len(phrase_vectors) > 0:
        return np.mean(phrase_vectors, axis=0)
    else:
        return np.zeros(vector_size)  # 如果没有词组向量，返回全零向量


def process_files(file_paths, glove_vectors, vector_size=50):
    file_vectors = []
    for file_path in file_paths:
        phrases = load_phrases_from_csv(file_path)
        file_vector = file_to_vector(phrases, glove_vectors, vector_size)
        file_vectors.append(file_vector)
    return file_vectors


if __name__ == "__main__":
    # 指定GloVe向量文件路径，加载50维的GloVe向量
    glove_file = "./glove.twitter.27B.50d.txt"
    glove_vectors = load_glove_vectors(glove_file)

    # 指定包含九个CSV文件的路径列表
    file_paths = ['../high_frequency_words/1_star_high_frequency.csv', '../high_frequency_words/2_star_high_frequency.csv', '../high_frequency_words/3_star_high_frequency.csv', '../high_frequency_words/4_star_high_frequency.csv',
                  '../high_frequency_words/5_star_high_frequency.csv', '../high_frequency_words/6_star_high_frequency.csv', '../high_frequency_words/7_star_high_frequency.csv', '../high_frequency_words/8_star_high_frequency.csv', '../high_frequency_words/9_star_high_frequency.csv']

    # 处理每个文件并生成向量
    file_vectors = process_files(file_paths, glove_vectors, vector_size=50)

    # 创建一个DataFrame来保存文件名称及其对应的向量
    file_names = [f'file{i + 1}' for i in range(len(file_vectors))]
    df = pd.DataFrame(file_vectors, index=file_names)

    # 将向量保存到CSV文件
    output_csv = 'file_vectors.csv'
    df.to_csv(output_csv, header=False)

    print(f"向量已保存到 {output_csv}")

