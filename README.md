# Breast Cancer Prediction - Machine Learning Assignment

## Project Overview
This repository contains a complete predictive decision support system for breast cancer classification.

The objective is to predict whether a breast tissue sample is:
- Benign
- Malignant

using machine learning models.

## Dataset
Dataset:
Breast Cancer Wisconsin (Original)

Dataset source:
https://archive.ics.uci.edu/dataset/15/breast

Kaggle source:
https://www.kaggle.com/datasets/saurabhbadole/breast-cancer-wisconsin-state

## Machine Learning Approach

Problem type:
Supervised Binary Classification

Input:
Nine tumour characteristics:
- Clump Thickness
- Uniformity of Cell Size
- Uniformity of Cell Shape
- Marginal Adhesion
- Single Epithelial Cell Size
- Bare Nuclei
- Bland Chromatin
- Normal Nucleoli
- Mitoses

Output:
- Benign
- Malignant

## Models Compared

1. Logistic Regression
- Interpretable linear classification model

2. Random Forest
- Nonlinear ensemble tree model

## Pipeline

1. Dataset audit
2. Missing value handling
3. Duplicate checking
4. Feature preparation
5. Model training
6. Cross-validation
7. Hyperparameter tuning
8. Final evaluation
9. Result visualization

## Validation

The project uses:
- Group-aware data splitting
- Cross-validation
- Untouched test evaluation

## Figures

All generated figures are available inside:

figures/

Included visualizations:
- Dataset quality summary
- Class distribution
- Missing values analysis
- Correlation heatmap
- Feature distributions
- Model comparison
- ROC curve
- Precision-recall curve
- Confusion matrix
- Final metrics
- Feature importance
- Calibration curve
- Error trade-off analysis

## Files

Root directory:

- breast_cancer_prediction.py
- breast_cancer_prediction.ipynb
- Breast_Cancer_Prediction_Report.docx
- README.md

Figures:

- figures/

## Assignment Coverage

This repository covers:

- Problem definition and framing
- Stakeholder analysis
- Dataset investigation
- Statistical analysis
- Model comparison
- Model validation
- Performance evaluation
- Conflict analysis
- Limitations
- Deployment considerations
- Reproducible machine learning workflow

## Scope Limitation

This model is a decision-support tool only.

It does not replace:
- Doctors
- Pathologists
- Clinical diagnosis
- Treatment decisions

External validation is required before real-world deployment.
