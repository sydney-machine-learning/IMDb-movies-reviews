import re
import os
import torch
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    pipeline
)

# --------------------------
# 配置设备
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ 当前设备: {device}")

# --------------------------
# 路径配置
# --------------------------
input_file = "imdb_reviews_tt14513804_captain.csv"
abuse_model_path = "../trained_model"
emotion_model_path = "../go_emotions_label_visualisation/go_emotions_trained_model"
output_file = input_file  # 直接覆盖原文件

# --------------------------
# 1️⃣ 辱骂检测模型
# --------------------------
abuse_model = DistilBertForSequenceClassification.from_pretrained(abuse_model_path)
abuse_tokenizer = DistilBertTokenizerFast.from_pretrained(abuse_model_path)
abuse_model.to(device).eval()

abuse_label_map = {0: "hateful", 1: "offensive", 2: "neither"}

def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", str(text))
    text = text.lower().strip()
    return re.sub(r'\s+', ' ', text)

def classify_abuse(text):
    text = preprocess_text(text)
    encoding = abuse_tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        outputs = abuse_model(**encoding)
    label_id = torch.argmax(outputs.logits, dim=-1).item()
    return abuse_label_map.get(label_id, "unknown")

# --------------------------
# 2️⃣ 情感分析（GoEmotions 多标签）
# --------------------------
emotion_model = DistilBertForSequenceClassification.from_pretrained(emotion_model_path)
emotion_tokenizer = DistilBertTokenizer.from_pretrained(emotion_model_path)
emotion_model.to(device).eval()

emotion_labels = [
    "Admiration", "Amusement", "Anger", "Annoyance", "Approval", "Caring", "Confusion",
    "Curiosity", "Desire", "Disappointment", "Disapproval", "Disgust", "Embarrassment",
    "Excitement", "Fear", "Gratitude", "Grief", "Joy", "Love", "Nervousness", "Optimism",
    "Pride", "Realization", "Relief", "Remorse", "Sadness", "Surprise", "Neutral"
]

def predict_emotions(text):
    inputs = emotion_tokenizer(str(text), return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = emotion_model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze().tolist()
    top_indices = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:3]
    return ", ".join([f"{emotion_labels[i]}({probs[i]:.4f})" for i in top_indices])

# --------------------------
# 3️⃣ Zero-Shot 多标签分类 (BART)
# --------------------------
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
zero_shot = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0 if torch.cuda.is_available() else -1)
zero_shot_labels = ["Storyline", "Acting", "Cinematography", "Soundtrack", "Rewatchability"]

def zero_shot_classify_batch(batch):
    results = zero_shot(batch["Review"], candidate_labels=zero_shot_labels, multi_label=True)
    return {
        "Target": [
            str({
                label: round(score, 4)
                for label, score in zip(res["labels"], res["scores"]) if score > 0.1
            }) for res in results
        ]
    }

# --------------------------
# 加载数据并处理
# --------------------------
df = pd.read_csv(input_file)
if "Review" not in df.columns:
    raise ValueError("❌ 找不到 'Review' 列")

# 1️⃣ 辱骂预测列
tqdm.pandas(desc="⛔ 预测 Abuse")
df["Abuse"] = df["Review"].progress_apply(classify_abuse)

# 2️⃣ 情感分析列
tqdm.pandas(desc="💬 预测 Sentiment")
df["Sentiment"] = df["Review"].progress_apply(predict_emotions)

# 3️⃣ 多标签主题分类列（Zero-Shot）
print("🎯 执行 Zero-Shot 分类...")
hf_dataset = Dataset.from_pandas(df)
hf_dataset = hf_dataset.map(zero_shot_classify_batch, batched=True, batch_size=16)
df = hf_dataset.to_pandas()

# 重新排序列，把三列放最前
ordered_cols = ["Abuse", "Sentiment", "Target"] + [col for col in df.columns if col not in ["Abuse", "Sentiment", "Target"]]
df = df[ordered_cols]

# 保存
df.to_csv(output_file, index=False)
print(f"✅ 所有预测完成，已保存覆盖文件：{output_file}")
