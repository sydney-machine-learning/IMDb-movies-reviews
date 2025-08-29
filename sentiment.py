from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

analyzer = SentimentIntensityAnalyzer()

star_ratings = []
avg_sentiment_intensity = []
std_sentiment_intensity = []

for star in range(1, 11):
    file_path = f'reviews_rating_{star}.csv'
    df = pd.read_csv(file_path)

    comments = df.iloc[:, 2].astype(str)
    sentiment_intensities = [abs(analyzer.polarity_scores(comment)['compound']) for comment in comments]

    avg_intensity = np.mean(sentiment_intensities)
    std_intensity = np.std(sentiment_intensities)

    star_ratings.append(star)
    avg_sentiment_intensity.append(avg_intensity)
    std_sentiment_intensity.append(std_intensity)

plt.figure(figsize=(10, 6))
plt.errorbar(star_ratings, avg_sentiment_intensity, yerr=std_sentiment_intensity, fmt='o-', capsize=5, label="Sentiment Intensity (VADER)")
plt.xlabel('Star Rating')
plt.ylabel('Average Sentiment Intensity')
plt.title('Sentiment Intensity by Star Rating (VADER)')
plt.axhline(0, color='gray', linestyle='--')
plt.legend()
plt.grid(True)
plt.show()
