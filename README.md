# News Article Classification with Machine Learning and Deep Learning Models

This repository presents a comprehensive study on classifying news articles into four categories: **Business**, **Entertainment**, **Health**, and **Technology**, using both traditional machine learning models and transformer-based models.

## Overview

In this project, we evaluate various text classification models, including Logistic Regression, Support Vector Machines (SVM), Naive Bayes, Random Forest, and K-Nearest Neighbors (KNN), alongside the transformer-based **DistilBERT** model. The goal is to compare their accuracy and computational efficiency on a Kaggle dataset of 33,288 news articles.

## Models Used

### Traditional Machine Learning Models:
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Naive Bayes**
- **Random Forest**
- **K-Nearest Neighbors (KNN)**

### Transformer Model:
- **DistilBERT** (Fine-tuned for classification)

## Data Preprocessing

- Data was scraped using the `newspaper3k` library, with additional handling for JavaScript-rendered content using `requests-html`.
- The text was preprocessed (lowercased, tokenized, lemmatized, and stopwords removed).
- Class imbalance was addressed using undersampling, resulting in a balanced dataset of 26,216 articles.

## Results

The table below summarizes the accuracy of different models using **TF-IDF** for feature extraction. The **DistilBERT** model achieves the highest accuracy but is computationally more expensive than traditional models like KNN with TF-IDF.

### Model Accuracy Comparison:

| **Model**                                    | **Accuracy**   |
|----------------------------------------------|----------------|
| KNN (TF-IDF, Cosine Distance, k=4)           | **96.834%**    |
| KNN (TF-IDF, Euclidean Distance, k=4)        | 96.777%        |
| Support Vector Machine (TF-IDF)              | 96.377%        |
| Logistic Regression (TF-IDF)                 | 95.67%         |
| Naive Bayes (TF-IDF)                         | 95.43%         |
| DistilBERT (Raw Text)                        | **96.834%**    |

## Key Findings

- **DistilBERT** provides the best performance in terms of accuracy, achieving **96.834%**, but its high computational cost makes it less efficient, especially on limited hardware.
- The **KNN** model with TF-IDF features and cosine distance also achieves **96.834%** accuracy, demonstrating that traditional models can be competitive when properly tuned, offering a more resource-efficient alternative.

## Conclusion

This study highlights the trade-offs between accuracy and computational efficiency in text classification. While transformer models like **DistilBERT** provide superior performance, traditional machine learning models such as **KNN** with **TF-IDF** can match or approach this performance while being far more efficient in terms of resources. This makes traditional models a viable option for deployment in resource-constrained environments.

## License

MIT License
