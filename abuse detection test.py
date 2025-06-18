import pandas as pd
import kagglehub
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# 1. 下载数据集到本地
path = kagglehub.dataset_download("mrmorj/hate-speech-and-offensive-language-dataset")
print("Path to dataset files:", path)

# 2. 读取 CSV 文件
# 数据集一般是单个 CSV 文件
csv_files = [f for f in os.listdir(path) if f.endswith('.csv')]
assert len(csv_files) > 0, "No CSV file found in the dataset directory."
dataset_path = os.path.join(path, csv_files[0])
df = pd.read_csv(dataset_path)

# 确认标签
# 0: hate speech, 1: offensive language, 2: neither
print(df['class'].value_counts())
print(df[['tweet', 'class']].head())

# 3. 划分训练/测试集（Hold-out）
train_df, eval_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df['class']
)


# 4. 自定义 PyTorch 数据集类
class HateSpeechDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer.encode_plus(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


# 5. 加载 tokenizer 和训练好的模型
model_path = "./trained_model"  # 你的模型文件夹路径
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

tokenizer = DistilBertTokenizer.from_pretrained("trained_model")
model = DistilBertForSequenceClassification.from_pretrained("trained_model")
model.eval()

# 6. 构建评估集 DataLoader
eval_dataset = HateSpeechDataset(
    texts=eval_df['tweet'].tolist(),
    labels=eval_df['class'].tolist(),
    tokenizer=tokenizer
)
eval_loader = DataLoader(eval_dataset, batch_size=32)

# 7. 执行模型评估
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in eval_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# 8. 输出分类报告
print(classification_report(
    all_labels, all_preds,
    target_names=['Hate Speech', 'Offensive Language', 'Neither']
))
