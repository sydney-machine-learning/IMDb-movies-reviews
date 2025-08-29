from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import emoji
import pandas as pd
import nltk
import re
import seaborn as sns
import matplotlib.pyplot as plt

# 读取 IMDb CSV 文件
file1 = '../Reviews/reviews_rating_1.csv'
file2 = '../Reviews/reviews_rating_2.csv'

df1 = pd.read_csv(file1, encoding='utf-8',encoding_errors="ignore")
df2 = pd.read_csv(file2,encoding='utf-8',encoding_errors="ignore")

nltk.download("stopwords")
nltk.download('wordnet')
# 假设评论列是 'review'
reviews1 = df1['Review'].tolist()
reviews2 = df2['Review'].tolist()

# 获取英语停用词
stopwords = stopwords.words('english')

# 保留的否定词和肯定词
negation_words = ['not', 'no', "don't", "doesn't", "didn't", "won't",
                  "isn't", "aren't", "wasn't", "weren't", 'never', 'none', 'nobody', 'nothing', 'neither']

affirmation_words = ['do', 'does', 'did', 'will', 'can', 'could', 'yes', 'yeah', 'sure', 'absolutely']

# 从停用词列表中移除这些否定和肯定词
for word in negation_words + affirmation_words:
    if word in stopwords:
        stopwords.remove(word)

# 添加自定义的停用词
stopwords.append('.')
stopwords.append('shall')
stopwords.append('thus')
stopwords.append("ive")

#stemmer = PorterStemmer()               #many mistakes(movi、movies、movie) influence bigrams and trigrams analysis
lemmatizer = WordNetLemmatizer()         #obtain more non synonymous and clearly defined phrases

# 数据清理函数
def clean_text(data):
    data = emoji.demojize(data)
    emoji_to_custom_text = {
        ":smiling_face_with_smiling_eyes:": "smiling_face",  # 将心眼笑脸替换为 "loves it"
    }

    for emoji_code, replacement in emoji_to_custom_text.items():
        data = data.replace(emoji_code, replacement)  # 替换表情符号为指定的文本
    data = data.lower()
    data = re.sub(r'[^\w\s]', '', data)
    data = ' '.join([word for word in data.split() if word.lower() not in stopwords])
    return data

def lemmatize_text(data):
    return ' '.join([lemmatizer.lemmatize(word) for word in data.split()])


# 对评论数据进行清理
cleaned_reviews1 = [clean_text(review) for review in reviews1]
cleaned_reviews2 = [clean_text(review) for review in reviews2]

stemmed_reviews1 = [lemmatize_text(review) for review in cleaned_reviews1]
stemmed_reviews2 = [lemmatize_text(review) for review in cleaned_reviews2]

# n-gram 可视化函数
def ngram_visualize_seaborn(data, n, title):
    wordList = re.sub("[^\w]", " ", data).split()
    ngrams_series = pd.Series(nltk.ngrams(wordList, n)).value_counts()[:50]

    # 将 n-grams 转换为字符串，避免 pandas 试图创建 MultiIndex
    ngrams_series.index = [' '.join(gram) for gram in ngrams_series.index]

    ngrams_df = pd.DataFrame({'phrase': ngrams_series.index, 'count': ngrams_series.values})

    # 绘制条形图
    plt.figure(figsize=(5, 5))
    sns.barplot(x=ngrams_df['count'], y=ngrams_df['phrase'], data=ngrams_df, color="blue")
    plt.title(title)
    plt.xlabel("Number of Occurrences")
    plt.ylabel(f"{n}-grams")
    plt.show()

# 汇总n-gram词组并保存到一个CSV文件的函数
def save_combined_ngrams_to_file(data, filename):
    combined_df = pd.DataFrame()

    # 对bigram, trigram和four-gram进行分析并保存
    for n in [2, 3, 4]:
        wordList = re.sub("[^\w]", " ", data).split()
        ngrams_series = pd.Series(nltk.ngrams(wordList, n)).value_counts()[:50]

        # 将n-grams转换为字符串，避免pandas创建MultiIndex
        ngrams_series.index = [' '.join(gram) for gram in ngrams_series.index]

        # 创建DataFrame，并标注n-gram类型
        ngrams_df = pd.DataFrame({'phrase': ngrams_series.index, 'count': ngrams_series.values})
        ngrams_df['n-gram_type'] = f"{n}-gram"

        # 汇总到一个DataFrame中
        combined_df = pd.concat([combined_df, ngrams_df])

    # 保存到文件
    combined_df.to_csv(filename, index=False)

# 示例：对文件1和文件2进行n-gram分析并将结果汇总保存
text1 = ' '.join(stemmed_reviews1)
save_combined_ngrams_to_file(text1, '1_star_high_frequency.csv')

text2 = ' '.join(stemmed_reviews2)
save_combined_ngrams_to_file(text2, '2_star_high_frequency.csv')

# 对文件1进行n-gram可视化
text1 = ' '.join(stemmed_reviews1)
for n in [2, 3, 4]:
    ngram_visualize_seaborn(text1, n, f'{n}-grams Distribution for 1-Star Reviews')

# 对文件2进行n-gram可视化
text2 = ' '.join(stemmed_reviews2)
for n in [2, 3, 4]:
    ngram_visualize_seaborn(text2, n, f'{n}-grams Distribution for 2-Star Reviews')
