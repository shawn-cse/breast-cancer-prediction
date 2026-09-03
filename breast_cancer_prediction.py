# ================================================================
# Predictive Decision Support Assignment
# Breast Cancer Wisconsin (Original) - Complete Modelling Pipeline
# ================================================================
# Purpose:
#   Predict malignant (positive class = 1) vs benign (0) tumors.
#   Compare TWO principled machine-learning models (Logistic Regression
#   and Random Forest), tune them without touching the test set, choose an
#   accuracy-optimised threshold subject to a malignant-sensitivity safety constraint,
#   and save publication-quality PNG evidence to /kaggle/working/.
#
# ================================================================

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedGroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
RANDOM_STATE = 42
# Assignment scenario constraint: missed malignant cases are costlier than false alarms.
# This is an analytical design choice for the assignment, NOT a clinical guideline.
MIN_SENSITIVITY = 0.95

DATA_PATH = (
    "/kaggle/input/datasets/saurabhbadole/"
    "breast-cancer-wisconsin-state/breast-cancer-wisconsin.data"
)
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLORS = {
    "navy": "#1F3B5C",
    "blue": "#3F6EA5",
    "teal": "#2A9D8F",
    "red": "#B33A3A",
    "gold": "#D4A72C",
    "grey": "#6B7280",
    "light": "#EEF2F6",
    "dark": "#111827",
}

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.titleweight": "bold",
    "axes.spines.top": False,
    "axes.spines.right": False,
})
sns.set_theme(style="whitegrid", context="notebook")


def out(name):
    return os.path.join(OUTPUT_DIR, name)


def save_current_figure(filename):
    plt.tight_layout()
    plt.savefig(out(filename), bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


def safe_div(a, b):
    return a / b if b else np.nan


def wilson_ci(successes, n, z=1.96):
    """95% Wilson interval for a proportion."""
    if n == 0:
        return (np.nan, np.nan)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt((p * (1 - p) / n) + (z**2 / (4 * n**2))) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def evaluate_probabilities(y_true, probabilities, threshold):
    pred = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    sensitivity = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    precision = safe_div(tp, tp + fp)
    npv = safe_div(tn, tn + fn)

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, pred),
        "Balanced Accuracy": balanced_accuracy_score(y_true, pred),
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision (PPV)": precision,
        "NPV": npv,
        "F1": f1_score(y_true, pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, pred),
        "ROC-AUC": roc_auc_score(y_true, probabilities),
        "PR-AUC": average_precision_score(y_true, probabilities),
        "Brier Score": brier_score_loss(y_true, probabilities),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def choose_threshold(y_true, probabilities):
    """
    Choose the probability threshold using ONLY training out-of-fold predictions.
    Because a missed malignant case is the higher-cost error in this scenario,
    first require sensitivity >= MIN_SENSITIVITY. Among feasible thresholds,
    maximise classification accuracy; tie-break by balanced accuracy, sensitivity,
    specificity, then threshold nearest 0.50. The holdout test set is never used.
    """
    thresholds = np.linspace(0.01, 0.99, 981)
    rows = []

    for threshold in thresholds:
        pred = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        sensitivity = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        accuracy = accuracy_score(y_true, pred)
        balanced = balanced_accuracy_score(y_true, pred)
        rows.append((threshold, accuracy, balanced, sensitivity, specificity, fn, fp))

    curve = pd.DataFrame(
        rows,
        columns=[
            "threshold", "accuracy", "balanced_accuracy", "sensitivity",
            "specificity", "false_negatives", "false_positives"
        ],
    )
    curve["distance_from_0_5"] = (curve["threshold"] - 0.50).abs()

    feasible = curve[curve["sensitivity"] >= MIN_SENSITIVITY].copy()
    # Fallback is defensive only; with this dataset feasible thresholds should exist.
    if feasible.empty:
        feasible = curve.copy()

    selected = feasible.sort_values(
        ["accuracy", "balanced_accuracy", "sensitivity", "specificity", "distance_from_0_5"],
        ascending=[False, False, False, False, True],
    ).iloc[0]

    return float(selected["threshold"]), curve


def save_table_png(df_table, filename, title, decimals=3, figsize=None):
    display_df = df_table.copy()
    for col in display_df.columns:
        if pd.api.types.is_numeric_dtype(display_df[col]):
            display_df[col] = display_df[col].map(
                lambda x: f"{x:.{decimals}f}" if pd.notna(x) else ""
            )

    if figsize is None:
        figsize = (max(9, 1.6 * len(display_df.columns)), max(3.2, 0.55 * len(display_df) + 2.0))

    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, loc="left", pad=16, color=COLORS["navy"], fontsize=14, fontweight="bold")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.45)

    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("#D1D5DB")
        if r == 0:
            cell.set_facecolor(COLORS["navy"])
            cell.get_text().set_color("white")
            cell.get_text().set_fontweight("bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F7F9FC")
        else:
            cell.set_facecolor("white")

    plt.savefig(out(filename), bbox_inches="tight", dpi=300, facecolor="white")
    plt.close()


# -----------------------------
# 2. LOAD DATA
# -----------------------------
columns = [
    "sample_code_number",
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
    "class",
]

feature_cols = [
    "clump_thickness",
    "uniformity_cell_size",
    "uniformity_cell_shape",
    "marginal_adhesion",
    "single_epithelial_cell_size",
    "bare_nuclei",
    "bland_chromatin",
    "normal_nucleoli",
    "mitoses",
]

pretty_feature_names = {
    "clump_thickness": "Clump thickness",
    "uniformity_cell_size": "Cell-size uniformity",
    "uniformity_cell_shape": "Cell-shape uniformity",
    "marginal_adhesion": "Marginal adhesion",
    "single_epithelial_cell_size": "Epithelial cell size",
    "bare_nuclei": "Bare nuclei",
    "bland_chromatin": "Bland chromatin",
    "normal_nucleoli": "Normal nucleoli",
    "mitoses": "Mitoses",
}

raw = pd.read_csv(
    DATA_PATH,
    header=None,
    names=columns,
    na_values=["?", " ?", "? "],
)

for c in columns:
    raw[c] = pd.to_numeric(raw[c], errors="coerce")

if not set(raw["class"].dropna().unique()).issubset({2, 4}):
    raise ValueError("Unexpected class labels. Expected only 2 (benign) and 4 (malignant).")

# Data-quality audit before modelling
original_rows = len(raw)
exact_duplicate_rows = int(raw.duplicated().sum())
missing_total = int(raw[feature_cols].isna().sum().sum())
repeated_id_rows = int(raw["sample_code_number"].duplicated().sum())
mixed_label_ids = int((raw.groupby("sample_code_number")["class"].nunique() > 1).sum())

# Remove exact duplicate records so identical rows do not receive extra weight.
df = raw.drop_duplicates().reset_index(drop=True)

# Positive class = malignant because false negatives are the high-risk error.
df["target"] = (df["class"] == 4).astype(int)

# ID is NOT a predictive feature. It is used only as a grouping variable.
X = df[feature_cols].copy()
y = df["target"].copy()
groups = df["sample_code_number"].copy()

print("\nDATASET AUDIT")
print("-------------")
print(f"Original rows: {original_rows}")
print(f"Rows after exact-duplicate removal: {len(df)}")
print(f"Exact duplicate rows removed: {exact_duplicate_rows}")
print(f"Missing predictor cells: {missing_total}")
print(f"Repeated-ID rows: {repeated_id_rows}")
print(f"IDs associated with both class labels: {mixed_label_ids}")
print("Class counts after deduplication:")
print(df["target"].value_counts().rename(index={0: "Benign", 1: "Malignant"}))

# -----------------------------
# 3. EDA / DATA-QUALITY PNGs
# -----------------------------
quality_rows = [
    ["Original records", original_rows],
    ["Records after exact-duplicate removal", len(df)],
    ["Exact duplicate records removed", exact_duplicate_rows],
    ["Unique sample IDs", df["sample_code_number"].nunique()],
    ["Repeated-ID rows in original data", repeated_id_rows],
    ["IDs with conflicting labels", mixed_label_ids],
    ["Missing predictor cells", missing_total],
    ["Benign records", int((y == 0).sum())],
    ["Malignant records", int((y == 1).sum())],
]
quality_df = pd.DataFrame(quality_rows, columns=["Data-quality indicator", "Value"])
save_table_png(
    quality_df,
    "data_quality_summary.png",
    "Data Quality Audit — Breast Cancer Wisconsin",
    decimals=0,
    figsize=(9, 6.2),
)

# Class distribution
counts = y.value_counts().sort_index()
fig, ax = plt.subplots(figsize=(6.5, 4.8))
bars = ax.bar(
    ["Benign", "Malignant"],
    [counts.get(0, 0), counts.get(1, 0)],
    color=[COLORS["teal"], COLORS["red"]],
    width=0.62,
)
ax.set_title("Class Distribution")
ax.set_ylabel("Number of records")
ax.set_xlabel("Diagnosis")
for b in bars:
    ax.text(
        b.get_x() + b.get_width() / 2,
        b.get_height() + max(counts) * 0.015,
        f"{int(b.get_height())}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )
ax.set_ylim(0, max(counts) * 1.12)
save_current_figure("class_distribution.png")

# Missing values
missing = raw[feature_cols].isna().sum().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8.5, 5.4))
labels = [pretty_feature_names[i] for i in missing.index]
ax.barh(labels, missing.values, color=COLORS["gold"])
ax.set_title("Missing Values by Predictor")
ax.set_xlabel("Missing records")
for i, v in enumerate(missing.values):
    ax.text(v + 0.15, i, str(int(v)), va="center")
save_current_figure("missing_values.png")

# Correlation matrix. Median fill is for visualization only; model imputation is inside CV pipelines.
eda_imputed = df[feature_cols].copy().fillna(df[feature_cols].median(numeric_only=True))
corr_df = eda_imputed.rename(columns=pretty_feature_names).copy()
corr_df["Malignant target"] = y.values
corr = corr_df.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(
    corr,
    cmap="vlag",
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Pearson correlation"},
    ax=ax,
)
ax.set_title("Feature Correlation Structure")
save_current_figure("correlation_heatmap.png")

# Distribution by diagnosis: 9 predictors in a journal-style panel
plot_df = eda_imputed.copy()
plot_df["Diagnosis"] = y.map({0: "Benign", 1: "Malignant"})
fig, axes = plt.subplots(3, 3, figsize=(13, 10.5))
for ax, col in zip(axes.flatten(), feature_cols):
    # Compatible with older Kaggle seaborn/matplotlib versions.
    # `legend=` is not supported by older seaborn boxplot APIs and can be
    # forwarded to Matplotlib's Axes.boxplot(), causing a TypeError.
    sns.boxplot(
        data=plot_df,
        x="Diagnosis",
        y=col,
        order=["Benign", "Malignant"],
        palette=[COLORS["teal"], COLORS["red"]],
        width=0.55,
        showfliers=True,
        ax=ax,
    )
    ax.set_title(pretty_feature_names[col], fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Score (1–10)")
fig.suptitle("Predictor Distributions by Diagnosis", fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig(out("feature_distributions.png"), bbox_inches="tight", dpi=300, facecolor="white")
plt.close()

# -----------------------------
# 4. LEAKAGE-SAFE TRAIN/TEST SPLIT
# -----------------------------
# One fold is held out as a final untouched test set.
# Grouping prevents records with the same sample code from appearing in both sets.
holdout_splitter = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)
train_idx, test_idx = next(holdout_splitter.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
g_train, g_test = groups.iloc[train_idx].copy(), groups.iloc[test_idx].copy()

assert set(g_train).isdisjoint(set(g_test)), "Group leakage detected between train and test sets."

split_summary = pd.DataFrame([
    ["Training records", len(X_train)],
    ["Test records", len(X_test)],
    ["Training malignant prevalence", y_train.mean()],
    ["Test malignant prevalence", y_test.mean()],
    ["Shared sample IDs across train/test", len(set(g_train).intersection(set(g_test)))],
], columns=["Validation indicator", "Value"])
save_table_png(
    split_summary,
    "validation_split_summary.png",
    "Leakage-Safe Holdout Design",
    decimals=3,
    figsize=(8.5, 4.2),
)

# -----------------------------
# 5. CANDIDATE MODELS
# -----------------------------
# The assignment requires comparison of at least two candidate approaches.
# We use exactly two complementary supervised machine-learning models:
#
# 1) Logistic Regression
#    - Interpretable baseline.
#    - Models a linear relationship between predictors and the log-odds of malignancy.
#    - Requires feature scaling because coefficient estimation is affected by feature scale.
#
# 2) Random Forest Classifier
#    - Nonlinear ensemble of decision trees trained on bootstrap samples.
#    - Captures interactions and nonlinear decision boundaries while reducing single-tree variance.
#    - Does not require standardisation because tree split rules depend on ordering, not scale.
#
# Using these two models directly supports the assignment's
# interpretability-versus-predictive-performance trade-off.

models = {
    "Logistic Regression": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=5000, solver="liblinear", random_state=RANDOM_STATE
            )),
        ]),
        "grid": {
            "model__C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
            "model__penalty": ["l1", "l2"],
            "model__class_weight": [None, "balanced"],
        },
    },
    "Random Forest": {
        "pipeline": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1)),
        ]),
        "grid": {
            "model__n_estimators": [300],
            "model__max_depth": [None, 5, 8],
            "model__min_samples_split": [2],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", 0.7],
            "model__class_weight": [None, "balanced"],
            "model__bootstrap": [True],
        },
    },
}


# Inner CV uses only the training set and keeps repeated sample IDs together.
inner_cv = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE + 7,
)

results = {}
comparison_rows = []

print("\nMODEL DEVELOPMENT")
print("-----------------")

for model_name, spec in models.items():
    print(f"\nTuning: {model_name}")

    search = GridSearchCV(
        estimator=spec["pipeline"],
        param_grid=spec["grid"],
        scoring={
            "accuracy": "accuracy",
            "roc_auc": "roc_auc",
            "pr_auc": "average_precision",
            "sensitivity": "recall",
            "balanced_accuracy": "balanced_accuracy",
            "f1": "f1",
        },
        refit="accuracy",
        cv=inner_cv,
        n_jobs=-1,
        return_train_score=True,
    )

    search.fit(X_train, y_train, groups=g_train)
    best_estimator = search.best_estimator_

    # Out-of-fold probabilities from the training set only.
    oof_prob = cross_val_predict(
        clone(best_estimator),
        X_train,
        y_train,
        groups=g_train,
        cv=inner_cv,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold, threshold_curve = choose_threshold(y_train, oof_prob)
    oof_metrics = evaluate_probabilities(y_train, oof_prob, threshold)

    # Fit final candidate on all training records, then evaluate on untouched test set.
    best_estimator.fit(X_train, y_train)
    test_prob = best_estimator.predict_proba(X_test)[:, 1]
    test_metrics = evaluate_probabilities(y_test, test_prob, threshold)

    best_index = search.best_index_
    cv_mean_accuracy = search.cv_results_["mean_test_accuracy"][best_index]
    cv_std_accuracy = search.cv_results_["std_test_accuracy"][best_index]
    cv_mean_auc = search.cv_results_["mean_test_roc_auc"][best_index]
    cv_std_auc = search.cv_results_["std_test_roc_auc"][best_index]
    cv_mean_pr = search.cv_results_["mean_test_pr_auc"][best_index]
    cv_std_pr = search.cv_results_["std_test_pr_auc"][best_index]
    cv_mean_f1 = search.cv_results_["mean_test_f1"][best_index]

    results[model_name] = {
        "search": search,
        "estimator": best_estimator,
        "best_params": search.best_params_,
        "oof_prob": oof_prob,
        "threshold": threshold,
        "threshold_curve": threshold_curve,
        "oof_metrics": oof_metrics,
        "test_prob": test_prob,
        "test_metrics": test_metrics,
        "cv_accuracy_mean": cv_mean_accuracy,
        "cv_accuracy_std": cv_std_accuracy,
        "cv_auc_mean": cv_mean_auc,
        "cv_auc_std": cv_std_auc,
        "cv_pr_mean": cv_mean_pr,
        "cv_pr_std": cv_std_pr,
    }

    comparison_rows.append({
        "Model": model_name,
        "CV Accuracy": cv_mean_accuracy,
        "CV Accuracy SD": cv_std_accuracy,
        "CV ROC-AUC": cv_mean_auc,
        "CV PR-AUC": cv_mean_pr,
        "CV F1": cv_mean_f1,
        "OOF threshold": threshold,
        "OOF Accuracy": oof_metrics["Accuracy"],
        "OOF sensitivity": oof_metrics["Sensitivity"],
        "OOF specificity": oof_metrics["Specificity"],
        "OOF balanced acc.": oof_metrics["Balanced Accuracy"],
    })

    print("Best parameters:", search.best_params_)
    print(f"CV accuracy: {cv_mean_accuracy:.4f} ± {cv_std_accuracy:.4f}")
    print(f"CV ROC-AUC: {cv_mean_auc:.4f} ± {cv_std_auc:.4f}")
    print(f"Accuracy-optimised OOF threshold: {threshold:.3f}")
    print(f"OOF sensitivity: {oof_metrics['Sensitivity']:.4f}")
    print(f"OOF specificity: {oof_metrics['Specificity']:.4f}")

comparison_df = pd.DataFrame(comparison_rows)

# -----------------------------
# 6. FINAL MODEL SELECTION
# -----------------------------
# IMPORTANT: selection is based on TRAINING/CROSS-VALIDATION performance only.
# The untouched holdout test set is NOT used to choose the model.
# Clinical-risk gate: require OOF sensitivity >= MIN_SENSITIVITY, then maximise CV accuracy.
# This preserves the user's accuracy objective while explicitly managing the higher cost of false negatives.
eligible_models = comparison_df[comparison_df["OOF sensitivity"] >= MIN_SENSITIVITY].copy()
if eligible_models.empty:
    eligible_models = comparison_df.copy()
final_model_name = eligible_models.sort_values(
    ["CV Accuracy", "OOF Accuracy", "CV ROC-AUC", "OOF sensitivity"],
    ascending=[False, False, False, False],
).iloc[0]["Model"]

final = results[final_model_name]
final_estimator = final["estimator"]
final_threshold = final["threshold"]
final_test_prob = final["test_prob"]
final_test_metrics = final["test_metrics"]

print("\nFINAL MODEL")
print("-----------")
print("Selected model:", final_model_name)
print(f"Decision threshold: {final_threshold:.3f}")
print(f"Selection rule: require OOF sensitivity >= {MIN_SENSITIVITY:.2f}, then highest mean training CV accuracy; tie-break by OOF accuracy, CV ROC-AUC, and sensitivity.")
print("Final test metrics:")
for key, value in final_test_metrics.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

# -----------------------------
# 7. MODEL COMPARISON PNGs
# -----------------------------
save_table_png(
    comparison_df,
    "model_comparison_table.png",
    "Candidate Model Comparison — Training/Cross-Validation Evidence",
    decimals=3,
    figsize=(14.0, 3.8),
)

# Grouped comparison plot
plot_metrics = comparison_df.set_index("Model")[[
    "CV Accuracy", "CV ROC-AUC", "CV PR-AUC", "OOF Accuracy",
    "OOF sensitivity", "OOF specificity"
]]
fig, ax = plt.subplots(figsize=(10.5, 5.8))
plot_metrics.plot(
    kind="bar",
    ax=ax,
    color=[COLORS["navy"], COLORS["blue"], COLORS["gold"], "#6D597A", COLORS["red"], COLORS["teal"]],
    width=0.76,
)
ax.set_ylim(0.70, 1.02)
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_title("Cross-Validated Candidate Model Performance")
ax.legend(frameon=True, ncol=2, loc="lower right")
ax.tick_params(axis="x", rotation=0)
save_current_figure("cv_model_comparison.png")

# Post-selection holdout ROC curves (reported, NOT used for selection)
fig, ax = plt.subplots(figsize=(7.2, 6.2))
roc_colors = [COLORS["navy"], COLORS["teal"]]
for (model_name, result), color in zip(results.items(), roc_colors):
    fpr, tpr, _ = roc_curve(y_test, result["test_prob"])
    auc = roc_auc_score(y_test, result["test_prob"])
    label = f"{model_name} (AUC={auc:.3f})"
    if model_name == final_model_name:
        label += " — selected"
    ax.plot(fpr, tpr, linewidth=2.3, color=color, label=label)
ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color=COLORS["grey"])
ax.set_xlabel("False positive rate (1 − specificity)")
ax.set_ylabel("True positive rate (sensitivity)")
ax.set_title("Holdout ROC Curves")
ax.legend(loc="lower right", frameon=True)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
save_current_figure("roc_curves.png")

# Precision-recall curves
fig, ax = plt.subplots(figsize=(7.2, 6.2))
for (model_name, result), color in zip(results.items(), roc_colors):
    precision, recall, _ = precision_recall_curve(y_test, result["test_prob"])
    ap = average_precision_score(y_test, result["test_prob"])
    label = f"{model_name} (AP={ap:.3f})"
    if model_name == final_model_name:
        label += " — selected"
    ax.plot(recall, precision, linewidth=2.3, color=color, label=label)
ax.axhline(y_test.mean(), linestyle="--", linewidth=1.2, color=COLORS["grey"],
           label=f"Malignant prevalence ({y_test.mean():.2f})")
ax.set_xlabel("Recall / sensitivity")
ax.set_ylabel("Precision / PPV")
ax.set_title("Holdout Precision–Recall Curves")
ax.legend(loc="lower left", frameon=True)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.02)
save_current_figure("precision_recall_curves.png")

# Holdout comparison table for transparent post-selection reporting.
# These test results are NOT used to select the winning model.
holdout_rows = []
for model_name, result in results.items():
    m = result["test_metrics"]
    holdout_rows.append({
        "Model": model_name,
        "Accuracy": m["Accuracy"],
        "Sensitivity": m["Sensitivity"],
        "Specificity": m["Specificity"],
        "F1": m["F1"],
        "ROC-AUC": m["ROC-AUC"],
        "PR-AUC": m["PR-AUC"],
        "Selected": "Yes" if model_name == final_model_name else "No",
    })
holdout_df = pd.DataFrame(holdout_rows).sort_values("Accuracy", ascending=False)
save_table_png(
    holdout_df,
    "holdout_model_comparison.png",
    "Untouched Holdout Performance — Reported After Model Selection",
    decimals=3,
    figsize=(13.5, 5.2),
)

# -----------------------------
# 8. THRESHOLD TRADE-OFF PNG
# -----------------------------
curve = final["threshold_curve"]
fig, ax = plt.subplots(figsize=(8.5, 5.7))
ax.plot(curve["threshold"], curve["accuracy"], color=COLORS["navy"], linewidth=2.6,
        label="Accuracy")
ax.plot(curve["threshold"], curve["sensitivity"], color=COLORS["red"], linewidth=2.1,
        label="Sensitivity")
ax.plot(curve["threshold"], curve["specificity"], color=COLORS["teal"], linewidth=2.1,
        label="Specificity")
ax.axvline(final_threshold, color=COLORS["navy"], linestyle=":", linewidth=2,
           label=f"Selected threshold = {final_threshold:.3f}")
ax.set_xlabel("Decision threshold for malignant probability")
ax.set_ylabel("Rate")
ax.set_ylim(0.75, 1.02)
ax.set_title(f"Accuracy and Error-Rate Threshold Trade-off — {final_model_name}")
ax.legend(loc="lower center", ncol=2, frameon=True)
save_current_figure("threshold_tradeoff.png")

# -----------------------------
# 9. FINAL CONFUSION MATRIX PNG
# -----------------------------
final_pred = (final_test_prob >= final_threshold).astype(int)
cm = confusion_matrix(y_test, final_pred, labels=[0, 1])
cm_total = cm.sum()
annotations = np.empty_like(cm).astype(object)
for i in range(2):
    for j in range(2):
        annotations[i, j] = f"{cm[i, j]}\n({cm[i, j] / cm_total:.1%} of test set)"

fig, ax = plt.subplots(figsize=(6.4, 5.4))
sns.heatmap(
    cm,
    annot=annotations,
    fmt="",
    cmap="Blues",
    cbar=False,
    linewidths=1,
    linecolor="white",
    square=True,
    xticklabels=["Predicted benign", "Predicted malignant"],
    yticklabels=["Actual benign", "Actual malignant"],
    ax=ax,
)
ax.set_title(f"Final Confusion Matrix — {final_model_name}")
ax.set_xlabel("")
ax.set_ylabel("")
save_current_figure("confusion_matrix.png")

# Final classification report as a journal-style table.
report_dict = classification_report(
    y_test,
    final_pred,
    target_names=["Benign", "Malignant"],
    output_dict=True,
    zero_division=0,
)
classification_df = pd.DataFrame(report_dict).T.reset_index().rename(columns={"index": "Class / average"})
classification_df = classification_df[["Class / average", "precision", "recall", "f1-score", "support"]]
save_table_png(
    classification_df,
    "classification_report.png",
    f"Classification Report — {final_model_name}",
    decimals=3,
    figsize=(10.5, 5.4),
)

# -----------------------------
# 10. FINAL METRICS PNGs
# -----------------------------
metrics_for_plot = {
    "Accuracy": final_test_metrics["Accuracy"],
    "Sensitivity": final_test_metrics["Sensitivity"],
    "Specificity": final_test_metrics["Specificity"],
    "Precision": final_test_metrics["Precision (PPV)"],
    "NPV": final_test_metrics["NPV"],
    "F1": final_test_metrics["F1"],
    "MCC": final_test_metrics["MCC"],
    "Balanced accuracy": final_test_metrics["Balanced Accuracy"],
    "ROC-AUC": final_test_metrics["ROC-AUC"],
    "PR-AUC": final_test_metrics["PR-AUC"],
}
fig, ax = plt.subplots(figsize=(10, 5.8))
metric_names = list(metrics_for_plot.keys())
metric_values = list(metrics_for_plot.values())
metric_colors = [
    COLORS["navy"], COLORS["red"], COLORS["teal"], COLORS["blue"], COLORS["gold"],
    "#6D597A", "#8B5E3C", "#457B9D", "#2A9D8F", "#7A5195"
]
bars = ax.bar(metric_names, metric_values, color=metric_colors[:len(metric_names)])
ax.set_ylim(0.75, 1.02)
ax.set_ylabel("Score")
ax.set_title(f"Final Holdout Performance — {final_model_name}")
ax.tick_params(axis="x", rotation=25)
for b, v in zip(bars, metric_values):
    ax.text(b.get_x() + b.get_width()/2, v + 0.006, f"{v:.3f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")
save_current_figure("final_metrics.png")

# Wilson 95% CI for proportion-type metrics
TN = final_test_metrics["TN"]
FP = final_test_metrics["FP"]
FN = final_test_metrics["FN"]
TP = final_test_metrics["TP"]

sens_ci = wilson_ci(TP, TP + FN)
spec_ci = wilson_ci(TN, TN + FP)
acc_ci = wilson_ci(TP + TN, TP + TN + FP + FN)
ppv_ci = wilson_ci(TP, TP + FP)
npv_ci = wilson_ci(TN, TN + FN)

summary_table = pd.DataFrame([
    ["Selected model", final_model_name, "", ""],
    ["Decision threshold", final_threshold, "", ""],
    ["Sensitivity", final_test_metrics["Sensitivity"], sens_ci[0], sens_ci[1]],
    ["Specificity", final_test_metrics["Specificity"], spec_ci[0], spec_ci[1]],
    ["Accuracy", final_test_metrics["Accuracy"], acc_ci[0], acc_ci[1]],
    ["Precision (PPV)", final_test_metrics["Precision (PPV)"], ppv_ci[0], ppv_ci[1]],
    ["NPV", final_test_metrics["NPV"], npv_ci[0], npv_ci[1]],
    ["F1 score", final_test_metrics["F1"], "", ""],
    ["Balanced accuracy", final_test_metrics["Balanced Accuracy"], "", ""],
    ["ROC-AUC", final_test_metrics["ROC-AUC"], "", ""],
    ["PR-AUC", final_test_metrics["PR-AUC"], "", ""],
    ["Brier score", final_test_metrics["Brier Score"], "", ""],
    ["False negatives", final_test_metrics["FN"], "", ""],
    ["False positives", final_test_metrics["FP"], "", ""],
], columns=["Indicator", "Value", "95% CI lower", "95% CI upper"])

# Convert only numeric-looking cells safely for a nice table
fig, ax = plt.subplots(figsize=(10.5, 7.6))
ax.axis("off")
ax.set_title(
    "Final Test-Set Evaluation Summary",
    loc="left",
    pad=16,
    color=COLORS["navy"],
    fontsize=14,
    fontweight="bold",
)
cell_text = []
for _, row in summary_table.iterrows():
    rendered = []
    for value in row:
        if isinstance(value, (float, np.floating)):
            rendered.append(f"{value:.3f}")
        else:
            rendered.append(str(value))
    cell_text.append(rendered)

tab = ax.table(
    cellText=cell_text,
    colLabels=summary_table.columns,
    cellLoc="center",
    colLoc="center",
    loc="center",
)
tab.auto_set_font_size(False)
tab.set_fontsize(9)
tab.scale(1, 1.45)
for (r, c), cell in tab.get_celld().items():
    cell.set_edgecolor("#D1D5DB")
    if r == 0:
        cell.set_facecolor(COLORS["navy"])
        cell.get_text().set_color("white")
        cell.get_text().set_fontweight("bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F7F9FC")
    else:
        cell.set_facecolor("white")
plt.savefig(out("final_metrics_table.png"), bbox_inches="tight", dpi=300, facecolor="white")
plt.close()

# -----------------------------
# 11. MODEL-AGNOSTIC FEATURE IMPORTANCE
# -----------------------------
# Permutation importance works for any selected model and measures the
# reduction in holdout ROC-AUC when a feature is randomly disrupted.
perm = permutation_importance(
    final_estimator,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=40,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
importance_df = pd.DataFrame({
    "Feature": [pretty_feature_names[c] for c in feature_cols],
    "Importance": perm.importances_mean,
    "SD": perm.importances_std,
}).sort_values("Importance", ascending=True)

fig, ax = plt.subplots(figsize=(8.5, 5.8))
ax.barh(
    importance_df["Feature"],
    importance_df["Importance"],
    xerr=importance_df["SD"],
    color=COLORS["blue"],
    alpha=0.9,
    capsize=3,
)
ax.axvline(0, color=COLORS["grey"], linewidth=1)
ax.set_xlabel("Mean decrease in holdout ROC-AUC after permutation")
ax.set_title(f"Permutation Feature Importance — {final_model_name}")
save_current_figure("feature_importance.png")

# -----------------------------
# 12. CALIBRATION PNG
# -----------------------------
prob_true, prob_pred = calibration_curve(
    y_test,
    final_test_prob,
    n_bins=6,
    strategy="quantile",
)
fig, ax = plt.subplots(figsize=(6.8, 6.0))
ax.plot([0, 1], [0, 1], "--", color=COLORS["grey"], linewidth=1.3, label="Perfect calibration")
ax.plot(prob_pred, prob_true, marker="o", markersize=6, linewidth=2.2,
        color=COLORS["navy"], label=final_model_name)
ax.set_xlabel("Mean predicted malignant probability")
ax.set_ylabel("Observed malignant proportion")
ax.set_title(f"Probability Calibration — {final_model_name}")
ax.legend(frameon=True, loc="upper left")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
text = f"Brier score = {final_test_metrics['Brier Score']:.3f}"
ax.text(0.04, 0.90, text, transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#D1D5DB"))
save_current_figure("calibration_curve.png")

# -----------------------------
# 13. ERROR COUNTS VS THRESHOLD PNG
# -----------------------------
fig, ax = plt.subplots(figsize=(8.5, 5.7))
ax.plot(curve["threshold"], curve["false_negatives"], color=COLORS["red"], linewidth=2.2,
        label="False negatives (missed malignant)")
ax.plot(curve["threshold"], curve["false_positives"], color=COLORS["teal"], linewidth=2.2,
        label="False positives (benign flagged)")
ax.axvline(final_threshold, color=COLORS["navy"], linestyle=":", linewidth=2,
           label=f"Selected threshold = {final_threshold:.3f}")
ax.set_xlabel("Decision threshold")
ax.set_ylabel("Out-of-fold error count")
ax.set_title("Operational Error Trade-off on Training Out-of-Fold Predictions")
ax.legend(frameon=True)
save_current_figure("error_tradeoff.png")

# -----------------------------
# 14. FINAL DECISION-SUPPORT SUMMARY PNG
# -----------------------------
summary_lines = [
    ["Selected model", final_model_name],
    ["Positive class", "Malignant tumor"],
    ["Selection evidence", "Training OOF only; test set untouched until final evaluation"],
    ["Model-selection rule", f"Sensitivity >= {MIN_SENSITIVITY:.2f}, then maximum mean CV accuracy"],
    ["Threshold rule", f"Sensitivity >= {MIN_SENSITIVITY:.2f}, then maximum OOF accuracy"],
    ["Final threshold", f"{final_threshold:.3f}"],
    ["Test accuracy", f"{final_test_metrics['Accuracy']:.3f}"],
    ["Test sensitivity", f"{final_test_metrics['Sensitivity']:.3f}"],
    ["Test specificity", f"{final_test_metrics['Specificity']:.3f}"],
    ["Test ROC-AUC", f"{final_test_metrics['ROC-AUC']:.3f}"],
    ["False negatives", str(final_test_metrics["FN"])],
    ["False positives", str(final_test_metrics["FP"])],
    ["Safe model role", "Decision support / prioritisation for clinician review"],
    ["Unsafe model role", "Independent diagnosis or treatment decision"],
]
summary_df = pd.DataFrame(summary_lines, columns=["Decision-support item", "Result / interpretation"])
save_table_png(
    summary_df,
    "decision_support_summary.png",
    "Decision-Support Interpretation",
    decimals=3,
    figsize=(13.5, 7.2),
)

# -----------------------------
# 15. PRINT OUTPUT FILE LIST
# -----------------------------
png_files = sorted([
    f for f in os.listdir(OUTPUT_DIR)
    if f.lower().endswith(".png")
])

print("\nPNG FILES SAVED TO /kaggle/working/")
print("-----------------------------------")
for f in png_files:
    print(" -", f)

print("\nASSIGNMENT INTERPRETATION NOTES")
print("-------------------------------")
print("1) Framing: supervised binary classification because class labels are available.")
print("2) Exactly two candidate ML models are compared, satisfying the assignment requirement of at least two approaches.")
print("3) Logistic Regression is the interpretable linear baseline; Random Forest is the nonlinear ensemble candidate.")
print("4) Malignant = positive class, so sensitivity measures the ability to avoid missed malignancies.")
print("5) ID is excluded from predictors and used as a grouping key to reduce leakage from repeated IDs.")
print("6) Missing values are median-imputed INSIDE each pipeline/fold, preventing preprocessing leakage.")
print("7) Scaling is applied to Logistic Regression because coefficient estimation is scale-sensitive.")
print("8) Random Forest is not scaled because tree split rules do not require standardised feature scales.")
print(f"9) Threshold/model selection requires OOF sensitivity >= {MIN_SENSITIVITY:.2f}, then maximises training CV/OOF accuracy; this manages the higher cost of false negatives.")
print("10) The untouched holdout test set is used once for final evaluation, not for model selection.")
print("11) Final output supports clinician review/prioritisation, NOT autonomous diagnosis or treatment.")
print("12) Before deployment, external validation on current, representative clinical data is required.")
