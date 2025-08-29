import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# **1. 设备检测 & 加载 BERT 预训练模型 / Device Check & Load BERT Pretrained Model**
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(device)


# **2. 计算情感强度函数 / Function to Calculate Sentiment Intensity**
def calculate_bert_sentiment(text):
    if not text.strip():
        return 0, 0  # 空评论返回 0 / Return 0 for empty reviews

    # Tokenization & Encoding
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    # 将输入张量移动到 GPU / Move input tensors to GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 预测 / Prediction
    with torch.no_grad():
        logits = model(**inputs).logits

    # 将 logits 移回 CPU，并计算 softmax 概率 / Move logits to CPU and calculate softmax probabilities
    scores = softmax(logits.cpu().numpy()[0])

    # 情感极性 (Positive - Negative)
    polarity_score = scores[2] - scores[0]

    # 情感强度（最高情感的绝对值）
    intensity_score = max(scores[0], scores[2])

    return polarity_score, intensity_score


# **3. 计算 10 个 CSV 文件的情感极性 & 强度 / Calculate Sentiment Polarity & Intensity for 10 CSV Files**
star_ratings = []
avg_polarity = []
std_polarity = []
avg_intensity = []
std_intensity = []

for star in range(1, 11):
    file_path = f'reviews_rating_{star}.csv'
    df = pd.read_csv(file_path)

    # 假设评论在第三列 / Assuming comments are in the third column
    comments = df.iloc[:, 2].astype(str)

    polarity_scores = []
    intensity_scores = []

    for comment in comments:
        polarity, intensity = calculate_bert_sentiment(comment)
        polarity_scores.append(polarity)
        intensity_scores.append(intensity)

    # 计算平均值 & 标准差 / Calculate mean & standard deviation
    avg_polarity.append(np.mean(polarity_scores))
    std_polarity.append(np.std(polarity_scores))
    avg_intensity.append(np.mean(intensity_scores))
    std_intensity.append(np.std(intensity_scores))
    star_ratings.append(star)

# **4. 绘制情感强度图（柱状图带误差条） / Plot Sentiment Intensity as Bar Chart with Error Bars**
plt.figure(figsize=(10, 6))

plt.bar(star_ratings, avg_intensity, yerr=std_intensity, capsize=10, width=0.65, color='skyblue',error_kw={'elinewidth': 3})

plt.xlabel('Rating', fontsize=28)  # 设置 x 轴标签及字体大小 / Set x-axis label and font size
plt.ylabel('Average Sentiment Intensity', fontsize=28)  # 设置 y 轴标签及字体大小 / Set y-axis label and font size

# 设置 x 轴显示 1-10 / Set x-axis ticks from 1 to 10
plt.xticks(range(1, 11), fontsize=28)
plt.yticks(fontsize=28)
plt.show()




