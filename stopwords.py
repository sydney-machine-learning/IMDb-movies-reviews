from nltk.corpus import stopwords

# 获取英语停用词
stopwrds = stopwords.words('english')

# 保留的否定词和肯定词
negation_words = ['not', 'no', "don't", "doesn't", "didn't", "won't",
                  "isn't", "aren't", "wasn't", "weren't", 'never', 'none', 'nobody', 'nothing', 'neither']

affirmation_words = ['do', 'does', 'did', 'will', 'can', 'could', 'yes', 'yeah', 'sure', 'absolutely']

# 从停用词列表中移除这些否定和肯定词
for word in negation_words + affirmation_words:
    if word in stopwrds:
        stopwrds.remove(word)

# 添加自定义的停用词
custom_stopwords = ['.', 'shall', 'thus']
stopwrds.extend(custom_stopwords)

# 打印最终的停用词列表
print(stopwrds)
