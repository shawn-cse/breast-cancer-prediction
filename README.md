# Breast Cancer Prediction

Machine learning project for breast cancer classification using the **Breast Cancer Wisconsin (Original) Dataset**.

The project predicts whether a breast tissue sample is:

- **Benign**
- **Malignant**

The project compares two supervised machine learning algorithms:

- Logistic Regression
- Random Forest

The complete workflow includes data analysis, preprocessing, model training, validation, evaluation, and visualization.

---

## Dataset

### Breast Cancer Wisconsin (Original)

Dataset Source:

UCI Machine Learning Repository:
https://archive.ics.uci.edu/dataset/15/breast

Kaggle Dataset:
https://www.kaggle.com/datasets/saurabhbadole/breast-cancer-wisconsin-state

### Dataset Information

- Original samples: 699
- Final samples after duplicate removal: 691
- Input features: 9
- Target variable: 1
- Problem type: Binary Classification

Target:

```
2 = Benign
4 = Malignant
```

---

## Input Features

The model uses the following tumour characteristics:

1. Clump Thickness
2. Uniformity of Cell Size
3. Uniformity of Cell Shape
4. Marginal Adhesion
5. Single Epithelial Cell Size
6. Bare Nuclei
7. Bland Chromatin
8. Normal Nucleoli
9. Mitoses

The sample ID is excluded from prediction.

---

## Machine Learning Models

### Logistic Regression

A linear classification model used as an interpretable baseline.

### Random Forest

An ensemble tree-based model used for learning nonlinear relationships and feature importance analysis.

---

## Machine Learning Workflow

```
Dataset
   |
Data Cleaning
   |
Missing Value Handling
   |
Feature Processing
   |
Model Training
   |
Cross Validation
   |
Model Comparison
   |
Performance Evaluation
   |
Prediction
```

---

## Data Preprocessing

The pipeline includes:

- Duplicate detection and removal
- Missing value handling
- Feature preparation
- Leakage-aware validation
- Model optimization

Missing values are handled using median imputation inside the machine learning pipeline.

---

## Evaluation Metrics

The models are evaluated using:

- Accuracy
- Sensitivity
- Specificity
- Precision
- F1 Score
- ROC-AUC
- PR-AUC
- Matthews Correlation Coefficient
- Confusion Matrix
- Calibration Analysis

---

# Visual Results

All generated figures are stored inside:

```
figures/
```

## Dataset Quality Summary

<img src="./figures/data_quality_summary.png" width="800">

## Class Distribution

<img src="./figures/class_distribution.png" width="800">

## Missing Value Analysis

<img src="./figures/missing_values.png" width="800">

## Correlation Heatmap

<img src="./figures/correlation_heatmap.png" width="800">

## Feature Distribution

<img src="./figures/feature_distributions.png" width="800">

## Validation Split

<img src="./figures/validation_split_summary.png" width="800">

## Model Comparison

<img src="./figures/model_comparison_table.png" width="800">

## Cross Validation Performance

<img src="./figures/cv_model_comparison.png" width="800">

## ROC Curve

<img src="./figures/roc_curves.png" width="800">

## Precision Recall Curve

<img src="./figures/precision_recall_curves.png" width="800">

## Threshold Analysis

<img src="./figures/threshold_tradeoff.png" width="800">

## Confusion Matrix

<img src="./figures/confusion_matrix.png" width="800">

## Final Metrics

<img src="./figures/final_metrics_table.png" width="800">

## Feature Importance

<img src="./figures/feature_importance.png" width="800">

## Calibration Curve

<img src="./figures/calibration_curve.png" width="800">

## Error Trade-off Analysis

<img src="./figures/error_tradeoff.png" width="800">

---

# Repository Structure

```
breast-cancer-prediction/

├── breast_cancer_prediction.py
├── breast_cancer_prediction.ipynb
├── README.md
│
└── figures/
    ├── data_quality_summary.png
    ├── class_distribution.png
    ├── confusion_matrix.png
    ├── roc_curves.png
    ├── feature_importance.png
    └── other visualization files
```

---

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Jupyter Notebook

---

# How to Run

Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Run Python script:

```bash
python breast_cancer_prediction.py
```

or open:

```
breast_cancer_prediction.ipynb
```

using Jupyter Notebook.

---

# Prediction Output

Input:

```
Breast tissue characteristics
```

Output:

```
Benign
or
Malignant
```

---

# Limitations

- The dataset is historical and limited in size.
- External validation is required before real-world deployment.
- The model should not replace medical professionals.
- Predictions should only support decision-making.

---

# Disclaimer

This project is developed for educational and research purposes only.

It is not a certified medical device and should not be used as a replacement for professional diagnosis, pathology assessment, or treatment decisions.
