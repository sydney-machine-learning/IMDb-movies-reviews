from datasets import load_dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
import torch

# === 1. 加载 GoEmotions 数据集 ===
dataset = load_dataset("go_emotions", "simplified")  # 使用简化标签版（单标签任务）

# 只取文本和情绪标签（label 是 0–27）
all_data = dataset['train']  # 全部数据（有 58k+ 样本）

# === 2. 取一部分作为测试集 ===
# 比如我们划分 20% 作为测试集
texts = all_data['text']
labels = all_data['labels']

# 过滤掉多标签样本（只保留单标签样本）
single_label_texts = []
single_label_targets = []
for text, label in zip(texts, labels):
    if len(label) == 1:  # 只保留单标签样本
        single_label_texts.append(text)
        single_label_targets.append(label[0])

# 划分为训练/测试集（这里只是划分评估集，训练模型你已有）
X_train, X_test, y_train, y_test = train_test_split(
    single_label_texts, single_label_targets, test_size=0.2, random_state=42, stratify=single_label_targets
)

# === 3. 加载本地训练好的模型 ===
model_path = "./go_emotions_trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

pipeline = TextClassificationPipeline(model=model, tokenizer=tokenizer, return_all_scores=False, device=0 if torch.cuda.is_available() else -1)

# === 4. 对测试集进行预测 ===
y_pred = []
for text in X_test:
    pred_label = pipeline(text)[0]['label']
    label_id = int(pred_label.replace("LABEL_", ""))
    y_pred.append(label_id)

# === 5. 输出评估指标 ===
accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='macro')
report = classification_report(y_test, y_pred, digits=4)

print(f"✅ Accuracy: {accuracy:.4f}")
print(f"✅ Macro Precision: {precision:.4f}")
print(f"✅ Macro Recall: {recall:.4f}")
print(f"✅ Macro F1 Score: {f1:.4f}")
print("📄 Classification Report:\n", report)
