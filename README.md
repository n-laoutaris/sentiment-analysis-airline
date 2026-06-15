# Airline Tweet Sentiment Analysis

A machine learning pipeline designed to classify public sentiment (positive, negative, neutral) toward US airlines using social media data.

![Project Visualization](./images/sample-output.png)

## Motivation & Project Context

In the media intelligence and monitoring industry, tracking public sentiment on platforms like X (formerly Twitter) is crucial for real-time brand reputation management. This project explores how to process unstructured social media text and evaluate classifiers both on accuracy and computational efficiency, a vital metric for real-time data flow systems.

## Engineering & Methodology
Beyond training models, this project emphasizes data science best practices:
- **Preventing Data Leakage:** Feature extraction is embedded directly within scikit-learn `Pipeline` objects. This ensures that the vocabulary is fitted only on the training folds during 5-fold cross-validation.
- **Dimensionality Trade-offs:** The pipeline evaluates the impact of feature space size on model performance and latency.
- **Evaluation Metrics:** Models are judged on Accuracy, Weighted F1-Score (to account for class imbalances), and Fit Time (computational cost).
- **Evaluating Data Manipulations:** The pipeline tests the impact of sparse-matrix standardization and aggressive regex/NLTK text pre-processing to determine if the computational gains outweigh the potential semantic loss.

## Tech Stack & Models
- **NLP Processing:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Classifiers Evaluated:** Logistic Regression, LinearSVC, Random Forest, Multilayer Perceptron (MLP)
- **Core Libraries:** `pandas`, `scikit-learn`, `matplotlib`

## Key Insights
By comparing the models, we observe that traditional linear models (Logistic Regression, LinearSVC) offer the best trade-off between high F1-scores and extremely low fit times, making them highly preferable for streaming media analysis compared to computationally heavy models like MLP or Random Forests. Aggressive stop-word removal degraded model accuracy, proving that standard "noise" in tweets actually carries vital semantic weight for sentiment analysis.

## Dataset

Source: [Kaggle - Twitter US Airline Sentiment](https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment)  
The dataset includes over 14,000 tweets about major US airlines, labeled as `positive`, `neutral`, or `negative`.
