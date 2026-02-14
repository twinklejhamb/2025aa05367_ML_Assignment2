# Machine Learning Assignment 2 – Heart Disease Classification

## Problem Statement
The objective of this project is to implement and evaluate multiple machine learning classification models to predict the presence of heart disease based on clinical attributes. The project also includes deploying the models using a Streamlit web application to demonstrate model performance interactively.

## Dataset Description
The dataset used is the Heart Disease Dataset (UCI Combined Version) obtained from Kaggle. It contains 1025 instances and 13 clinical features such as age, chest pain type, cholesterol level, and maximum heart rate. The target variable is binary, indicating the presence or absence of heart disease.

## Models Used and Performance Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.8098 | 0.9298 | 0.7619 | 0.9143 | 0.8312 | 0.6309 |
| Decision Tree | 0.9854 | 0.9857 | 1.0000 | 0.9714 | 0.9855 | 0.9712 |
| KNN | 0.8634 | 0.9629 | 0.8738 | 0.8571 | 0.8654 | 0.7269 |
| Naive Bayes | 0.8293 | 0.9043 | 0.8070 | 0.8762 | 0.8402 | 0.6602 |
| Random Forest (Ensemble) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| XGBoost (Ensemble) | 0.9610 | 0.9885 | 0.9709 | 0.9524 | 0.9615 | 0.9221 |



## Model-wise Observations

| ML Model Name | Observation about model performance |
|---------------|--------------------------------------|
| Logistic Regression | Logistic Regression achieved strong recall (0.9143) and high AUC (0.9298), indicating good ability to correctly identify heart disease cases. However, its overall accuracy (0.8098) and MCC (0.6309) are lower compared to tree-based and ensemble models, suggesting limited ability to capture complex non-linear patterns. |
| Decision Tree | Decision Tree achieved extremely high performance across all metrics, including accuracy (0.9854) and MCC (0.9712). While this indicates strong predictive capability, such near-perfect performance may suggest overfitting, as single decision trees are prone to memorizing training patterns. |
| KNN | KNN showed balanced and stable performance with good AUC (0.9629) and moderate MCC (0.7269). Its results demonstrate that distance-based methods can perform well after proper feature scaling, though performance remains lower than ensemble methods. |
| Naive Bayes | Naive Bayes achieved reasonable performance with accuracy (0.8293) and AUC (0.9043). However, its assumption of feature independence likely limited its predictive strength compared to more flexible models like Decision Trees and ensembles. |
| Random Forest (Ensemble) | Random Forest achieved perfect scores (1.0000) across all evaluation metrics, indicating excellent classification performance. The ensemble approach effectively captured complex feature interactions and significantly reduced overfitting compared to a single Decision Tree. |
| XGBoost (Ensemble) | XGBoost achieved very high accuracy (0.9610), AUC (0.9885), and MCC (0.9221), demonstrating strong generalization performance. Its gradient boosting framework effectively modeled complex relationships while maintaining robustness. |
