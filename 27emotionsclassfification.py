import os
import torch
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from tqdm import tqdm

# 选择 GPU 或 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 模型路径
MODEL_PATH = "./go_emotions_trained_model"

# 加载 DistilBERT 预训练模型和 tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)  # ✅ 把模型迁移到 GPU
model.eval()  # ✅ 设置为评估模式

# **正确的情感标签顺序（从你提供的数据提取）**
EMOTION_LABELS = [
    "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", "Confusion",
    "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust", "Embarrassment",
    "Excitement", "Fear", "Gratitude", "Grief", "Joy", "Love", "Nervousness", "Optimism",
    "Pride", "Realization", "Relief", "Remorse", "Sadness", "Surprise", "Neutral"
]

# 预测函数：获取 **至少 3 个概率最高的情感标签**
def predict_emotions(review_text):
    """对单条 Review 进行情感分类预测，返回至少 3 个最高概率的情感标签"""
    inputs = tokenizer(review_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # ✅ 确保所有输入在 GPU

    with torch.no_grad():
        outputs = model(**inputs)  # ✅ 在 GPU 上计算
    logits = outputs.logits
    probabilities = torch.sigmoid(logits).squeeze().tolist()  # ✅ 计算概率（多标签）

    # 获取 **概率最高的 3 个情感标签**
    top_indices = sorted(range(len(probabilities)), key=lambda i: probabilities[i], reverse=True)[:3]
    predicted_labels = [f"{EMOTION_LABELS[i]}({probabilities[i]:.4f})" for i in top_indices]

    return predicted_labels


# 处理所有 CSV 文件
for rating in range(1, 11):
    input_file = f"../Reviews/reviews_rating_{rating}.csv"  # 输入文件
    output_file = f"predicted_reviews_rating_{rating}.csv"  # ✅ 现在直接保存在当前目录

    print(f"Processing: {input_file}")

    # 读取 CSV 文件
    if not os.path.exists(input_file):
        print(f"Skipping {input_file}, file not found.")
        continue

    df = pd.read_csv(input_file)

    # 确保列名正确
    if "Review" not in df.columns:
        print(f"Skipping {input_file}, 'Review' column not found.")
        continue

    # 对每条评论进行情感预测
    tqdm.pandas(desc=f"Predicting emotions for {input_file}")
    df["Predicted_Emotions"] = df["Review"].progress_apply(predict_emotions)

    # ✅ 保存到当前目录
    df.to_csv(output_file, index=False)
    print(f"Saved results to {output_file}")

print("All files processed successfully!")
