import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from collections import Counter
import nltk
from nltk.corpus import stopwords

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stop_words.add('im')

# Load CSV files (classified_file1.csv to classified_file10.csv) into a single DataFrame
file_list = [f'classified_file{i}.csv' for i in range(1, 11)]
df_list = [pd.read_csv(file) for file in file_list]
df = pd.concat(df_list, ignore_index=True)

# Text cleaning function: convert to lowercase, remove special characters, and extra spaces
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Apply text cleaning on the 'Review' column
df['cleaned_review'] = df['Review'].apply(clean_text)

# Function for n-grams analysis with stopwords removal
def ngrams_analysis(text_series, n=2, top_n=5):
    all_ngrams = []
    for text in text_series:
        tokens = word_tokenize(text)
        # Remove stopwords
        tokens = [token for token in tokens if token not in stop_words]
        all_ngrams.extend(list(ngrams(tokens, n)))
    return Counter(all_ngrams).most_common(top_n)

# Helper function to convert n-grams counter to a DataFrame for plotting
def ngrams_to_df(ngrams_counter):
    df_ngrams = pd.DataFrame(ngrams_counter, columns=['ngram', 'count'])
    df_ngrams['ngram'] = df_ngrams['ngram'].apply(lambda x: ' '.join(x))
    return df_ngrams

# -------------------- Analysis & Visualization for predicted_class 0 --------------------
class0_reviews = df[df['predicted_class'] == 0]['cleaned_review']

# Compute top 5 bigrams and top 5 trigrams for class 0
bigrams_class0 = ngrams_analysis(class0_reviews, n=2, top_n=5)
trigrams_class0 = ngrams_analysis(class0_reviews, n=3, top_n=5)

# Convert results to DataFrames
df_bigrams_class0 = ngrams_to_df(bigrams_class0)
df_trigrams_class0 = ngrams_to_df(trigrams_class0)

# Add a column to indicate n-gram type
df_bigrams_class0['ngram_type'] = 'bigram'
df_trigrams_class0['ngram_type'] = 'trigram'

# Combine bigrams and trigrams DataFrames
df_combined_class0 = pd.concat([df_bigrams_class0, df_trigrams_class0])

plt.figure(figsize=(12, 6))
palette = {'bigram': 'dodgerblue', 'trigram': 'darkorange'}
ax = sns.barplot(x='count', y='ngram', data=df_combined_class0, hue='ngram_type',
                 dodge=False, palette=palette)

# 设置坐标轴标签及字体大小
plt.xlabel('Frequency', fontsize=28)
plt.ylabel('N-grams', fontsize=28)
# 将 y 轴刻度标签字体减小，减少横向空间占用
ax.tick_params(axis='y', labelsize=28)
ax.tick_params(axis='x', labelsize=28)
plt.legend(title='N-gram Type', fontsize=28, title_fontsize=28)
plt.tight_layout()
plt.show()

# -------------------- Analysis & Visualization for predicted_class 1 --------------------
class1_reviews = df[df['predicted_class'] == 1]['cleaned_review']

# Compute top 5 bigrams and top 5 trigrams for class 1
bigrams_class1 = ngrams_analysis(class1_reviews, n=2, top_n=5)
trigrams_class1 = ngrams_analysis(class1_reviews, n=3, top_n=5)

# Convert results to DataFrames
df_bigrams_class1 = ngrams_to_df(bigrams_class1)
df_trigrams_class1 = ngrams_to_df(trigrams_class1)

# Add n-gram type column
df_bigrams_class1['ngram_type'] = 'bigram'
df_trigrams_class1['ngram_type'] = 'trigram'

# Combine DataFrames
df_combined_class1 = pd.concat([df_bigrams_class1, df_trigrams_class1])

plt.figure(figsize=(12, 6))
ax = sns.barplot(x='count', y='ngram', data=df_combined_class1, hue='ngram_type',
                 dodge=False, palette=palette)

plt.xlabel('Frequency', fontsize=28)
plt.ylabel('N-grams', fontsize=28)
# 同样将 y 轴刻度标签字体减小
ax.tick_params(axis='y', labelsize=28)
ax.tick_params(axis='x', labelsize=28)
plt.legend(title='N-gram Type', fontsize=28, title_fontsize=28)
plt.tight_layout()
plt.show()

