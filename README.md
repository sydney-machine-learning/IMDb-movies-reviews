# IMDb Movie Reviews Analysis

## Project Overview
The IMDb-movies-reviews project achieves its goal of analyzing a large volume of movie reviews and drawing analytical conclusions by concatenating an algorithm based on N-grams and cosine similarity with three fine-tuned language models. It addresses the following questions:

1. Why did this movie succeed/fail?

 2. Why did this movie perform poorly at the box office despite high ratings (low ratings, high box office)?

3. What trajectory of change do English-speaking audiences hope to see in future films?
 

## Directory Structure

├── IMDb movie reviews analysis
│ ├── 4cases/ # Using a comprehensive technical framework, we analyzed the four selected films and obtained the results.
│ ├── High_grequency_words/ # Used to convert reviews into text vectors and predict ratings based on cosine similarity.
│ ├── Go_Emotions_label_visualisation/ # Results obtained from analyzing movie reviews after fine-tuning the model using the GoEmotions dataset.
│ ├── Processed data and plots/ #Raw data with labels.
│ └── Raw data and analysis/ #Reviews extracted from IMDb website.
