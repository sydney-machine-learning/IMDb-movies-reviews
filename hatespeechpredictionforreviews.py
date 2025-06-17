import os
import re
import pandas as pd
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

# --------------------------
# 数据预处理函数 / Data Preprocessing Function
# --------------------------
def preprocess_text(text):
    """
    清理文本中的特殊符号、标点，保留字母、数字和空格
    Clean text by removing special characters and punctuation, keeping only letters, numbers, and spaces.
    """
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = text.lower().strip()
    return re.sub(r'\s+', ' ', text)

# --------------------------
# 加载模型和分词器 / Load the Model and Tokenizer
# --------------------------
model_path = "./trained_model"  # 替换为你的模型文件夹路径 / Replace with your model folder path
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# --------------------------
# 定义预测函数 / Define Prediction Function
# --------------------------
def classify_review(text):
    # 对文本进行预处理 / Preprocess the text
    text = preprocess_text(text)
    # 使用分词器对文本进行编码 / Encode the text using the tokenizer
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=256, return_tensors="pt")
    encoding = {key: val.to(device) for key, val in encoding.items()}
    with torch.no_grad():
        outputs = model(**encoding)
    # 获取预测的标签 / Get the predicted label
    predicted_label = torch.argmax(outputs.logits, dim=-1).item()
    return predicted_label

# --------------------------
# 批量处理 CSV 文件 / Process Multiple CSV Files
# --------------------------
input_dir = "./Reviews"  # CSV 文件所在目录 / Directory containing CSV files
for i in range(1, 11):
    csv_file = os.path.join(input_dir, f"reviews_rating_{i}.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        # 假设 CSV 文件中第三列的列名为 "Review" / Assuming the third column is named "Review"
        df['predicted_class'] = df['Review'].apply(classify_review)
        output_file = f"classified_file{i}.csv"
        df.to_csv(output_file, index=False)
        print(f"文件 {csv_file} 已分类，结果保存在 {output_file}")
    else:
        print(f"文件 {csv_file} 不存在。")
