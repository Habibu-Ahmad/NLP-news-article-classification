# News Article Classification Project

This project implements a news article classification system to predict categories such as *business*, *entertainment*, *health*, and *technology*. It explores various machine learning models and their performance using two common feature extraction techniques: **TF-IDF** and **Bag-of-Words (BoW)**. The models evaluated include traditional machine learning algorithms and transformer-based models, specifically **DistilBERT**.

## Overview

The goal of this project is to build a robust classification system for news articles using different models, such as **Logistic Regression**, **Support Vector Machines (SVM)**, **Naive Bayes**, **K-Nearest Neighbors (KNN)**, **Random Forest**, and **DistilBERT**. The system was tested on a publicly available Kaggle dataset with the aim of comparing the performance of traditional machine learning algorithms versus a transformer-based model (DistilBERT). 

The project explores how various models perform in terms of classification accuracy and computational efficiency, considering the constraints of running models like DistilBERT on platforms like Google Colab free tier.

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

\begin{table}[h!]
\centering
\caption{Model Accuracy Comparison}
\begin{tabular}{lcc}
\toprule
\textbf{Model} & \textbf{BoW Accuracy} & \textbf{TF-IDF Accuracy} \\
\midrule
Logistic Regression      & 93.58\% & 93.67\% \\
Support Vector Machine   & 92.64\% & 94.13\% \\
Naive Bayes              & 93.52\% & 93.39\% \\
Random Forest            & 87.93\% & 88.36\% \\
KNN (Cosine Distance)    & 94.66\% & 96.83\% \\
KNN (Euclidean Distance) & 88.95\% & 94.67\% \\
DistilBERT (Transformer) & -       & 96.834\% \\
\bottomrule
\end{tabular}
\end{table}

- **KNN** with **TF-IDF** and **Cosine distance** yielded the highest accuracy among traditional models, achieving **96.83%** (for **k=4**).
- The **DistilBERT** model achieved an impressive accuracy of **96.834%** but was computationally more expensive to run.

## Conclusion

This project demonstrates the strengths of both traditional machine learning models and deep learning models (such as DistilBERT) for news article classification. The **KNN** model with **TF-IDF** offers a compelling trade-off between high accuracy and computational efficiency. On the other hand, **DistilBERT** provides state-of-the-art accuracy but is more resource-intensive, making it less practical for real-time applications on platforms with limited resources.

## Installation

1. Clone this repository:
