
import re
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments

# 设备检测 / Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 设置 PyTorch 显存优化
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 释放显存（避免碎片化）
torch.cuda.empty_cache()

# --------------------------
# 数据预处理函数 / Data Preprocessing Function
# --------------------------
def preprocess_text(text):
    """清理文本中的特殊符号、标点，保留字母、数字和空格"""
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # 只保留字母、数字和空格
    text = text.lower().strip()  # 转换为小写，去除两端空格
    return re.sub(r'\s+', ' ', text)  # 去除多余的空格


# --------------------------
# 自定义 Dataset 类 / Custom Dataset Class
# --------------------------
class AbuseDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):  # ✅ 降低 max_length 以减少显存
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze() for key, val in encoding.items()}
        item['labels'] = torch.tensor(self.labels[index], dtype=torch.long)
        return item


def main():
    # --------------------------
    # 1. 数据加载与预处理 / Data Loading and Preprocessing
    # --------------------------
    df = pd.read_csv('labeled_data.csv')  # 修改为你的数据集文件路径
    df['tweet'] = df['tweet'].apply(preprocess_text)

    texts = df['tweet'].tolist()
    labels = df['class'].tolist()

    # ✅ 确保 train/test 拆分时类别均衡，避免数据不均匀导致的 loss 震荡
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    # 初始化 Tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')

    # ✅ 降低 max_length 以减少显存
    train_dataset = AbuseDataset(train_texts, train_labels, tokenizer, max_length=256)
    val_dataset = AbuseDataset(val_texts, val_labels, tokenizer, max_length=256)

    # --------------------------
    # 2. 模型初始化与训练设置 / Model Initialization and Training Setup
    # --------------------------
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased', num_labels=3).to(device)

    training_args = TrainingArguments(
        fp16=True,  # ✅ 开启 FP16 以减少 50% 显存
        output_dir='./results',
        num_train_epochs=5,
        per_device_train_batch_size=2,  # ✅ 降低 batch_size
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,  # ✅ 使用梯度累积，等效于 batch_size=8
        warmup_steps=500,
        learning_rate=3e-5,
        weight_decay=0.005,
        eval_strategy="epoch",
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    # --------------------------
    # 3. 模型训练与保存 / Model Training and Saving
    # --------------------------
    trainer.train()

    model.save_pretrained('./trained_model')
    tokenizer.save_pretrained('./trained_model')

    # --------------------------
    # 4. 使用训练好的模型进行预测 / Prediction Using the Trained Model
    # --------------------------
    my_df = pd.read_csv('./Reviews/reviews_rating_1.csv')
    my_df['Review'] = my_df['Review'].apply(preprocess_text)

    texts_to_predict = my_df['Review'].tolist()

    # ✅ 释放显存，避免 OOM
    torch.cuda.empty_cache()

    # 确保模型在 GPU 上
    model.to(device)
    model.eval()

    # ✅ 降低 max_length，减少显存占用
    encodings = tokenizer(texts_to_predict, truncation=True, padding=True, max_length=256, return_tensors='pt')

    # ✅ 确保数据在 GPU 上
    encodings = {key: val.to(device) for key, val in encodings.items()}

    # 预测
    with torch.no_grad():
        outputs = model(**encodings)

    # 获取预测类别
    predictions = torch.argmax(outputs.logits, dim=-1).cpu().numpy()  # ✅ 先转回 CPU，避免 NumPy 计算错误

    # 保存结果
    my_df['predicted_class'] = predictions
    my_df.to_csv('predicted_texts.csv', index=False)
    print("预测结果已保存至 predicted_texts.csv")


if __name__ == '__main__':
    main()
