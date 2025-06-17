from transformers import pipeline
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# 加载预训练的情感分析模型
#classifier = pipeline("sentiment-analysis", model="nlptown/bert-base-multilingual-uncased-sentiment")
# 使用 Twitter 数据的情感分析模型
classifier = pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")

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

# 将文本转换为 GloVe 向量
def text_to_glove_vector(text, glove_vectors, vector_size=50):
    words = text.split()  # 按空格分词
    word_vectors = [glove_vectors[word] for word in words if word in glove_vectors]

    if len(word_vectors) == 0:
        return np.zeros(vector_size)  # 如果没有找到词，返回全零向量

    return np.mean(word_vectors, axis=0)

def load_vectors_from_csv(file_path):
    df = pd.read_csv(file_path, header=None)
    vectors = df.values.astype(np.float32)  # 将整个CSV文件的值转换为float32类型
    return vectors

def find_most_similar(glove_vector, csv_vectors):
    similarities = cosine_similarity([glove_vector], csv_vectors)[0]  # 计算余弦相似度
    most_similar_index = np.argmax(similarities)  # 找到最大相似度的索引
    return most_similar_index, similarities[most_similar_index]

# 从CSV文件加载9个50维的向量
csv_file = "./GloVe_cosine_similarity/file_vectors.csv"  # 请确保CSV文件路径正确
csv_vectors = load_vectors_from_csv(csv_file)

# 加载 GloVe 向量
glove_file = "./GloVe_cosine_similarity/glove.twitter.27B.50d.txt"  # 请确保GloVe文件路径正确
glove_vectors = load_glove_vectors(glove_file)



while True:
    # 获取用户输入的文本
    text = input("Please input('exit' is over): ")

    # 如果输入是“退出”，则跳出循环结束程序
    if text.lower() == 'exit':
        print("Program is over.")
        break
    # 对输入文本进行情感分析
    results = classifier([text])
    sentiment = results[0]['label']


    # 输出情感分析结果
    print(f"txt: {text}")
    print(f"Sentiment Label: {sentiment}, Confidence Score: {results[0]['score']:.4f}\n")

    # 将文本转换为 GloVe 向量
    glove_vector = text_to_glove_vector(text, glove_vectors)
    print(f"GloVe Vector: {glove_vector}\n")

    # 计算余弦相似度并找出最相似的行
    #most_similar_index, similarity_score = find_most_similar(glove_vector, csv_vectors)
    #print(f"Rating: {most_similar_index + 1} star，Similarity Score: {similarity_score:.4f}\n")

    if sentiment == 'negative':
        selected_vectors = csv_vectors[:4]  # 选择前四行
        original_indices = list(range(4))  # 对应原始文件中的第1到4行
    elif sentiment == 'neutral':
        selected_vectors = csv_vectors[3:6]  # 选择第4到6行
        original_indices = list(range(4, 7))  # 对应原始文件中的第4到6行
    elif sentiment == 'positive':
        selected_vectors = csv_vectors[-4:]  # 选择最后四行
        original_indices = list(range(len(csv_vectors) - 4, len(csv_vectors)))  # 对应原始文件的最后4行

        # 计算余弦相似度并找出最相似的行
    most_similar_index, similarity_score = find_most_similar(glove_vector, selected_vectors)
    most_similar_original_index = original_indices[most_similar_index]  # 转换为原始文件中的行号

    print(
        f"Most similar to {most_similar_original_index + 1} star")



















