# News Article Classification Project.

## Overview

The goal of this project is to build a robust classification system for news articles using different models, such as **Logistic Regression**, **Support Vector Machines (SVM)**, **Naive Bayes**, **K-Nearest Neighbors (KNN)**, **Random Forest**, and **DistilBERT**. The system was tested on a publicly available Kaggle dataset with the aim of comparing the performance of traditional machine learning algorithms versus a transformer-based model (DistilBERT). 

## Data Preprocessing

- Data was scraped using the `newspaper3k` library, with additional handling for JavaScript-rendered content using `requests-html`.
- The text was preprocessed (lowercased, tokenized, lemmatized, and stopwords removed).
- Class imbalance was addressed using undersampling, resulting in a balanced dataset of 26,216 articles.


## Models & Techniques

1. **Traditional Models**:
    - Logistic Regression
    - Support Vector Machine (SVM)
    - Naive Bayes
    - Random Forest
    - K-Nearest Neighbors (KNN) with cosine and Euclidean distances
  
2. **Deep Learning Model**:
    - **DistilBERT** fine-tuned on raw article content

The project uses **TF-IDF** and **BoW** for text vectorization and evaluates model performance across multiple metrics, including accuracy, precision, recall, and F1-score.

## Results

The results highlight the performance of the models with respect to their classification accuracy, and the following table summarizes the **accuracy** for the key models evaluated in the project:

### Model Accuracy Comparison:

| **Model**                                    | **Accuracy**   |
|----------------------------------------------|----------------|
| KNN (TF-IDF, Cosine Distance, k=4)           | **96.834%**    |
| KNN (TF-IDF, Euclidean Distance, k=4)        | 96.777%        |
| Support Vector Machine (TF-IDF)              | 96.377%        |
| Logistic Regression (TF-IDF)                 | 95.67%         |
| Naive Bayes (TF-IDF)                         | 95.43%         |
| Random Forest                                | 90.25%
| DistilBERT (Raw Text)                        | **96.834%**    |


- **KNN** with **TF-IDF** and **Cosine distance** yielded the highest accuracy among traditional models, achieving **96.83%** (for **k=4**).
- The **DistilBERT** model achieved an impressive accuracy of **96.834%** but was computationally more expensive to run.

## Conclusion

This project demonstrates the strengths of both traditional machine learning models and deep learning models (such as DistilBERT) for news article classification. The **KNN** model with **TF-IDF** offers a compelling trade-off between high accuracy and computational efficiency. On the other hand, **DistilBERT** provides state-of-the-art accuracy but is more resource-intensive, making it less practical for real-time applications on platforms with limited resources.

## Installation

1. Clone this repository:
