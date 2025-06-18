import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# List of CSV filenames
csv_files = [f'classified_file{i}.csv' for i in range(1, 11)]

# Initialize an empty list to store reviews with predicted_class == 0
negative_reviews = []

# Read each file and extract reviews where predicted_class is 0
for file in csv_files:
    df = pd.read_csv(file)
    filtered_reviews = df[df['predicted_class'] == 2]['Review'].astype(str).tolist()
    negative_reviews.extend(filtered_reviews)

# Combine all extracted reviews into a single text string
text = " ".join(negative_reviews)

# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(text)

# Plot the word cloud
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()