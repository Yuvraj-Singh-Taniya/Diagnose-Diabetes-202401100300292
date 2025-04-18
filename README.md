# Diabetes Diagnosis Using Logistic Regression

## Project Overview

This project focuses on building a supervised machine learning model to predict whether a patient has diabetes based on diagnostic measurements. The dataset used is the well-known Pima Indians Diabetes Dataset, which contains information on various medical parameters for female patients of Pima Indian heritage aged 21 and older.

The goal is to classify individuals as diabetic or non-diabetic, helping to support early medical intervention and health decision-making.

---

## Problem Statement

Diabetes is a chronic condition that affects how the body processes blood sugar (glucose). Early diagnosis is crucial for effective management and reducing complications. Using machine learning, we aim to predict the presence of diabetes (1 for diabetic, 0 for non-diabetic) using features such as:

- Number of pregnancies
- Glucose concentration
- Blood pressure
- Skin thickness
- Insulin levels
- Body Mass Index (BMI)
- Diabetes Pedigree Function (a function which scores likelihood of diabetes based on family history)
- Age

This binary classification task is important for developing tools to assist healthcare providers in diagnosis and resource allocation.

---

## Methodology

1. **Data Loading and Exploration**:
   - Load dataset using Pandas.
   - Examine dataset structure and check for missing/null values.

2. **Preprocessing**:
   - Split dataset into features (`X`) and target (`y`).
   - Perform an 80-20 train-test split using stratification to maintain class distribution.
   - Standardize features using `StandardScaler` for better model performance.

3. **Model Building**:
   - Train a Logistic Regression model on the training set.

4. **Model Evaluation**:
   - Evaluate the model on the test set using metrics like:
     - Accuracy
     - Precision
     - Recall
     - F1-Score
     - Confusion Matrix
     - Classification Report

5. **Visualization**:
   - Generate a confusion matrix heatmap using Seaborn to visually interpret model predictions.

---

## Libraries Used

| Library            | Purpose                                           |
|--------------------|---------------------------------------------------|
| `pandas`           | Data loading and manipulation                     |
| `matplotlib.pyplot`| Plotting graphs and figures                       |
| `seaborn`          | Creating attractive statistical plots (heatmap)   |
| `scikit-learn`     | Machine learning utilities, preprocessing, models |

---

## Algorithm Used: Logistic Regression

Logistic Regression is a linear model used for binary classification tasks. It calculates the probability of a data point belonging to a particular class using the logistic (sigmoid) function. In this context, it models the probability that a patient has diabetes given their medical measurements.

It was chosen for this problem due to its:
- Interpretability
- Efficiency with small-to-medium datasets
- Suitability for binary outcomes

Hyperparameters:
- `max_iter = 200` to ensure convergence during training

---

## Results and Evaluation



**Explanation**:
- True Negatives (87): Correctly predicted as non-diabetic
- False Positives (13): Predicted diabetic but are not
- False Negatives (18): Predicted non-diabetic but are diabetic
- True Positives (36): Correctly predicted as diabetic

### Evaluation Metrics

| Metric     | Value   |
|------------|---------|
| Accuracy   | 0.798    |
| Precision  | 0.7347   |
| Recall     | 0.6667   |
| F1 Score   | 0.6981   |

### Classification Report

          precision    recall  f1-score   support
accuracy                           0.80       154

**Interpretation**:
- The model performs slightly better at predicting non-diabetic individuals.
- The precision and recall values indicate a balanced performance across classes.

---

## Dataset Reference

- **Dataset**: Pima Indians Diabetes Dataset
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/diabetes)

---

## Conclusion

This project demonstrates how logistic regression can be effectively used to classify individuals as diabetic or non-diabetic based on medical measurements. While performance is reasonable, future improvements could involve:
- Trying more advanced models like Random Forest or XGBoost
- Handling potential outliers or missing values
- Performing feature selection or engineering
- Applying cross-validation for more reliable evaluation


