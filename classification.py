import pandas as pd
import os
import torch
from transformers import pipeline
from datasets import Dataset

# **关闭 symlinks 警告**
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# **确保使用 GPU**
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=device)

# **电影评论的基础分类标签**
categories = ["Storyline", "Acting", "Cinematography", "Soundtrack", "Rewatchability"]

# **要处理的文件列表**
file_names = [f"classified_file{i}.csv" for i in range(1, 11)]


# **批量分类函数**
def batch_classify(examples):
    results = classifier(examples["Review"], candidate_labels=categories, multi_label=True)

    # 只保留概率 > 0.3 的类别，并转换为字符串格式
    target_list = [
        {label: round(score, 4) for label, score in zip(result["labels"], result["scores"]) if score > 0.3}
        for result in results
    ]

    return {"Target": [str(target) for target in target_list]}  # **转换为字符串**


# **遍历文件进行处理**
for file_name in file_names:
    if os.path.exists(file_name):
        print(f"🔄 处理文件: {file_name} ...")

        # **加载 CSV 文件**
        df = pd.read_csv(file_name)

        # **转换为 Hugging Face Dataset**
        dataset = Dataset.from_pandas(df)

        # **批量处理（提高 GPU 利用率）**
        dataset = dataset.map(batch_classify, batched=True, batch_size=16)

        # **转换回 Pandas DataFrame**
        df = dataset.to_pandas()

        # **确保 `Target` 列已更新，并保存为字符串**
        df["Target"] = df["Target"].astype(str)

        # **保存分类后的文件**
        output_file = f"processed_{file_name}"
        df.to_csv(output_file, index=False)
        print(f"✅ 处理完成，结果保存到: {output_file}\n")

print("🚀 所有文件处理完成！")
