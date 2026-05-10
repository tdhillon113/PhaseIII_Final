"""
Cancer Gene Expression Classifier
==================================
Breast cancer subtype classification using gene expression data (GSE45827).
Models: Kernel SVM, Random Forest, Logistic Regression, Naive Bayes

Dataset: Breast_GSE45827.csv
Expected columns: 'samples', 'type', and gene expression feature columns (_at suffix).

Usage:
    python main.py --data path/to/Breast_GSE45827.csv
    python main.py  # defaults to Breast_GSE45827.csv in the current directory
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; use "TkAgg" or "Qt5Agg" for live windows

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.model_selection import (
    train_test_split, StratifiedKFold,
    GridSearchCV, RandomizedSearchCV,
    learning_curve, validation_curve,
    cross_validate, cross_val_score,
)
from sklearn.metrics import (
    f1_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, make_scorer,
)

np.random.seed(42)

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def save_and_show(filename: str) -> None:
    """Save the current figure and print confirmation."""
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [saved] {filename}")


def separator(title: str = "", width: int = 65) -> None:
    print("\n" + "=" * width)
    if title:
        print(f"  {title}")
        print("=" * width)


# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────────────────────

def load_data(csv_path: str) -> pd.DataFrame:
    separator("1. LOADING DATA")
    df = pd.read_csv(csv_path)
    print(f"  File  : {csv_path}")
    print(f"  Shape : {df.shape}  ({df.shape[0]} samples × {df.shape[1]} columns)")
    print(f"  Columns preview: {list(df.columns[:5])} ...")
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ──────────────────────────────────────────────────────────────────────────────

def eda(df: pd.DataFrame) -> None:
    separator("2. EXPLORATORY DATA ANALYSIS")

    print("\n  Data Types:\n", df.dtypes.value_counts().to_string())

    missing = df.isnull().sum().sum()
    print(f"\n  Total Missing Values: {missing}  ({'none found' if missing == 0 else 'ATTENTION'})")

    print("\n  Class Distribution (type column):")
    print(df["type"].value_counts().to_string())

    # ── Class distribution bar chart ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    df["type"].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Class Distribution – Cancer Subtypes", fontweight="bold")
    ax.set_xlabel("Cancer Subtype")
    ax.set_ylabel("Sample Count")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    save_and_show("eda_class_distribution.png")

    # ── Summary statistics (numeric columns) ──────────────────────────────────
    print("\n  Summary Statistics (first 5 gene columns):")
    gene_cols = [c for c in df.columns if c not in ("samples", "type")]
    print(df[gene_cols[:5]].describe().round(4).to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    separator("3. PREPROCESSING")

    # Feature / label split
    X = df.drop(["samples", "type"], axis=1).values.astype(np.float64)
    y_raw = df["type"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    class_names = le.classes_
    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  Class counts: {dict(zip(class_names, np.bincount(y)))}")

    # Train / test split (stratified, 70/30)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"\n  Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # Variance filtering (remove near-constant features)
    stds = np.std(X_train, axis=0)
    nonzero_mask = stds > 0.01
    X_train_filt = X_train[:, nonzero_mask]
    X_test_filt  = X_test[:, nonzero_mask]
    print(f"  Features after variance filtering: {X_train_filt.shape[1]:,}  "
          f"(removed {(~nonzero_mask).sum():,})")

    return X_train_filt, X_test_filt, y_train, y_test, class_names, nonzero_mask, df


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

def build_pipelines_and_grids():
    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=0.95, svd_solver="full")),
        ("clf",    SVC(kernel="rbf", class_weight="balanced",
                       cache_size=500, random_state=42)),
    ])
    svm_grid = {
        "clf__C":     [0.1, 1, 10],
        "clf__gamma": [0.001, 0.01, "scale"],
    }

    rf_pipe = Pipeline([
        ("clf", RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=42)),
    ])
    rf_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_features": ["sqrt", "log2"],
    }

    lr_pipe = Pipeline([
        ("select", SelectKBest(f_classif, k=1000)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=10000, tol=0.01,
            class_weight="balanced", solver="saga", random_state=42)),
    ])
    lr_grid = {
        "clf__C":       [0.01, 0.1, 1, 10],
        "clf__penalty": ["l1", "l2"],
    }

    nb_pipe = Pipeline([
        ("select", SelectKBest(f_classif, k=1000)),
        ("clf",    GaussianNB()),
    ])
    nb_grid = {
        "select__k":          [500, 1000, 2000],
        "clf__var_smoothing":  np.logspace(-9, -3, 4),
    }

    return (svm_pipe, svm_grid, rf_pipe, rf_grid,
            lr_pipe, lr_grid, nb_pipe, nb_grid)


# ──────────────────────────────────────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING
# ──────────────────────────────────────────────────────────────────────────────

def tune_models(X_train_filt, y_train,
                svm_pipe, svm_grid,
                rf_pipe, rf_grid,
                lr_pipe, lr_grid,
                nb_pipe, nb_grid):

    scoring = "balanced_accuracy"
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    separator("5. HYPERPARAMETER TUNING")

    # ── SVM ───────────────────────────────────────────────────────────────────
    print("\n  [5a] Kernel SVM  (9 combos × 5 folds = 45 fits)")
    svm_search = GridSearchCV(svm_pipe, svm_grid, cv=cv,
                               scoring=scoring, n_jobs=-1, verbose=0)
    svm_search.fit(X_train_filt, y_train)
    print(f"       Best C     : {svm_search.best_params_['clf__C']}")
    print(f"       Best Gamma : {svm_search.best_params_['clf__gamma']}")
    print(f"       Best CV Balanced Acc : {svm_search.best_score_:.4f}")

    # ── Random Forest ─────────────────────────────────────────────────────────
    print("\n  [5b] Random Forest  (6 combos × 5 folds = 30 fits)")
    rf_search = GridSearchCV(rf_pipe, rf_grid, cv=cv,
                              scoring=scoring, n_jobs=-1, verbose=0)
    rf_search.fit(X_train_filt, y_train)
    print(f"       Best n_estimators : {rf_search.best_params_['clf__n_estimators']}")
    print(f"       Best max_features : {rf_search.best_params_['clf__max_features']}")
    print(f"       Best CV Balanced Acc : {rf_search.best_score_:.4f}")

    # ── Logistic Regression ───────────────────────────────────────────────────
    print("\n  [5c] Logistic Regression  (8 combos × 5 folds = 40 fits)")
    lr_search = GridSearchCV(lr_pipe, lr_grid, cv=cv,
                              scoring=scoring, n_jobs=-1, verbose=0)
    lr_search.fit(X_train_filt, y_train)
    print(f"       Best C       : {lr_search.best_params_['clf__C']}")
    print(f"       Best Penalty : {lr_search.best_params_['clf__penalty']}")
    print(f"       Best CV Balanced Acc : {lr_search.best_score_:.4f}")

    # ── Naive Bayes ───────────────────────────────────────────────────────────
    print("\n  [5d] Naive Bayes  (randomized, 8 iters × 5 folds = 40 fits)")
    nb_search = RandomizedSearchCV(nb_pipe, nb_grid, n_iter=8, cv=cv,
                                    scoring=scoring, n_jobs=-1,
                                    random_state=42, verbose=0)
    nb_search.fit(X_train_filt, y_train)
    print(f"       Best var_smoothing : {nb_search.best_params_['clf__var_smoothing']:.2e}")
    print(f"       Best K             : {nb_search.best_params_['select__k']}")
    print(f"       Best CV Balanced Acc : {nb_search.best_score_:.4f}")

    return svm_search, rf_search, lr_search, nb_search, cv


# ──────────────────────────────────────────────────────────────────────────────
# 6. LEARNING CURVES
# ──────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(svm_search, rf_search, lr_search, nb_search,
                          X_train_filt, y_train, cv):
    separator("6. LEARNING CURVES")

    scoring = "balanced_accuracy"
    entries = [
        (svm_search.best_estimator_, "Kernel SVM",          "lc_svm.png"),
        (rf_search.best_estimator_,  "Random Forest",       "lc_rf.png"),
        (lr_search.best_estimator_,  "Logistic Regression", "lc_lr.png"),
        (nb_search.best_estimator_,  "Naive Bayes",         "lc_nb.png"),
    ]

    for estimator, title, filename in entries:
        train_sizes = np.linspace(0.1, 1.0, 8)
        train_sz, train_scores, val_scores = learning_curve(
            estimator, X_train_filt, y_train,
            train_sizes=train_sizes, cv=cv,
            scoring=scoring, n_jobs=-1,
        )
        train_mean, train_std = train_scores.mean(1), train_scores.std(1)
        val_mean,   val_std   = val_scores.mean(1),   val_scores.std(1)

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(train_sz, train_mean, "o-", color="steelblue",  label="Training score")
        ax.plot(train_sz, val_mean,   "o-", color="darkorange", label="Validation score")
        ax.fill_between(train_sz, train_mean - train_std, train_mean + train_std,
                         alpha=0.15, color="steelblue")
        ax.fill_between(train_sz, val_mean - val_std, val_mean + val_std,
                         alpha=0.15, color="darkorange")
        ax.set_title(f"Learning Curve – {title}", fontweight="bold")
        ax.set_xlabel("Training samples")
        ax.set_ylabel("Balanced Accuracy")
        ax.legend(loc="best")
        plt.tight_layout()
        save_and_show(filename)


# ──────────────────────────────────────────────────────────────────────────────
# 7. VALIDATION CURVES
# ──────────────────────────────────────────────────────────────────────────────

def plot_validation_curves(svm_search, rf_search, lr_search, nb_search,
                            X_train_filt, y_train, cv):
    separator("7. VALIDATION CURVES")

    scoring = "balanced_accuracy"

    def _plot(estimator, param_name, param_range, title, filename, log=True):
        train_scores, val_scores = validation_curve(
            estimator, X_train_filt, y_train,
            param_name=param_name, param_range=param_range,
            cv=cv, scoring=scoring, n_jobs=-1,
        )
        train_mean, train_std = train_scores.mean(1), train_scores.std(1)
        val_mean,   val_std   = val_scores.mean(1),   val_scores.std(1)
        x_axis = np.log10(param_range) if log else np.arange(len(param_range))

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_axis, train_mean, "o-", color="steelblue",  label="Training score")
        ax.plot(x_axis, val_mean,   "o-", color="darkorange", label="Validation score")
        ax.fill_between(x_axis, train_mean - train_std, train_mean + train_std,
                         alpha=0.15, color="steelblue")
        ax.fill_between(x_axis, val_mean - val_std, val_mean + val_std,
                         alpha=0.15, color="darkorange")
        ax.set_title(f"Validation Curve – {title}", fontweight="bold")
        ax.set_xlabel(f"log10({param_name})" if log else param_name)
        ax.set_ylabel("Balanced Accuracy")
        ax.legend(loc="best")
        plt.tight_layout()
        save_and_show(filename)

    _plot(svm_search.best_estimator_, "clf__C",
          np.array([0.001, 0.01, 0.1, 1, 10, 100]),
          "SVM – C", "vc_svm_C.png")

    _plot(lr_search.best_estimator_, "clf__C",
          np.array([0.001, 0.01, 0.1, 1, 10]),
          "Logistic Regression – C", "vc_lr_C.png")

    _plot(rf_search.best_estimator_, "clf__n_estimators",
          np.array([100, 200, 500]),
          "Random Forest – n_estimators", "vc_rf_nest.png", log=False)

    _plot(nb_search.best_estimator_, "clf__var_smoothing",
          np.logspace(-9, -3, 4),
          "Naive Bayes – var_smoothing", "vc_nb_vs.png")


# ──────────────────────────────────────────────────────────────────────────────
# 8. CROSS-VALIDATION EVALUATION
# ──────────────────────────────────────────────────────────────────────────────

def cross_validation_evaluation(svm_search, rf_search, lr_search, nb_search,
                                  X_train_filt, y_train):
    separator("8. FULL CROSS-VALIDATION EVALUATION (5-Fold Stratified)")
    print("  Every sample used as test data across folds")

    cv_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1":          make_scorer(f1_score, average="macro"),
    }
    models_cv = {
        "Kernel SVM":          svm_search.best_estimator_,
        "Random Forest":       rf_search.best_estimator_,
        "Logistic Regression": lr_search.best_estimator_,
        "Naive Bayes":         nb_search.best_estimator_,
    }

    cv_results = {}
    for name, model in models_cv.items():
        scores = cross_validate(
            model, X_train_filt, y_train,
            cv=cv_full, scoring=cv_scoring,
            n_jobs=-1, return_train_score=True,
        )
        r = {
            "Train Balanced Acc":   np.round(scores["train_balanced_accuracy"].mean(), 4),
            "CV Test Balanced Acc": np.round(scores["test_balanced_accuracy"].mean(),  4),
            "CV Test Bal Acc Std":  np.round(scores["test_balanced_accuracy"].std(),   4),
            "Train Macro F1":       np.round(scores["train_macro_f1"].mean(),          4),
            "CV Test Macro F1":     np.round(scores["test_macro_f1"].mean(),           4),
            "CV Test F1 Std":       np.round(scores["test_macro_f1"].std(),            4),
        }
        cv_results[name] = r
        print(f"\n  {name}")
        print(f"  Train Balanced Acc  : {r['Train Balanced Acc']}")
        print(f"  CV Test Balanced Acc: {r['CV Test Balanced Acc']} "
              f"(+/- {r['CV Test Bal Acc Std']})")
        print(f"  Train Macro F1      : {r['Train Macro F1']}")
        print(f"  CV Test Macro F1    : {r['CV Test Macro F1']} "
              f"(+/- {r['CV Test F1 Std']})")
        print("  " + "-" * 50)

    return cv_results


# ──────────────────────────────────────────────────────────────────────────────
# 9. RESULTS TABLE
# ──────────────────────────────────────────────────────────────────────────────

def results_table(svm_search, rf_search, lr_search, nb_search,
                   X_train_filt, X_test_filt, y_train, y_test):
    separator("9. RESULTS TABLE — Train | Validation (CV) | Test")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "Kernel SVM":          svm_search.best_estimator_,
        "Random Forest":       rf_search.best_estimator_,
        "Logistic Regression": lr_search.best_estimator_,
        "Naive Bayes":         nb_search.best_estimator_,
    }

    table_data = []
    row_labels  = []

    print(f"\n  {'Model':<22} {'Train Bal/F1':<18} {'CV Bal/F1':<18} {'Test Bal/F1'}")
    print("  " + "-" * 75)

    for name, model in models.items():
        y_tr_pred = model.predict(X_train_filt)
        train_bal = balanced_accuracy_score(y_train, y_tr_pred)
        train_f1  = f1_score(y_train, y_tr_pred, average="macro")

        val_bal = cross_val_score(model, X_train_filt, y_train,
                                   cv=cv, scoring="balanced_accuracy", n_jobs=-1).mean()
        val_f1  = cross_val_score(model, X_train_filt, y_train,
                                   cv=cv, scoring="f1_macro",          n_jobs=-1).mean()

        y_te_pred = model.predict(X_test_filt)
        test_bal  = balanced_accuracy_score(y_test, y_te_pred)
        test_f1   = f1_score(y_test, y_te_pred, average="macro")

        row_labels.append(name)
        table_data.append([
            f"{train_bal:.4f} / {train_f1:.4f}",
            f"{val_bal:.4f}  / {val_f1:.4f}",
            f"{test_bal:.4f} / {test_f1:.4f}",
        ])
        print(f"  {name:<22} {train_bal:.4f}/{train_f1:.4f}        "
              f"{val_bal:.4f}/{val_f1:.4f}        "
              f"{test_bal:.4f}/{test_f1:.4f}")

    # ── Table figure ──────────────────────────────────────────────────────────
    col_labels = ["Train\n(Bal Acc / Macro F1)",
                  "Validation CV\n(Bal Acc / Macro F1)",
                  "Test\n(Bal Acc / Macro F1)"]

    fig, ax = plt.subplots(figsize=(13, 3))
    ax.axis("off")
    tbl = ax.table(cellText=table_data, rowLabels=row_labels,
                   colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.3, 2.2)

    for j in range(3):
        tbl[(0, j)].set_facecolor("#2c3e50")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, 5):
        color = "#eaf0fb" if i % 2 == 0 else "white"
        for j in range(-1, 3):
            tbl[(i, j)].set_facecolor(color)

    plt.title("Model Performance: Train | Validation | Test\n"
              "(Balanced Accuracy / Macro F1)",
              fontsize=12, pad=15, fontweight="bold")
    plt.tight_layout()
    save_and_show("results_table_final.png")


# ──────────────────────────────────────────────────────────────────────────────
# 10. CV BAR CHART
# ──────────────────────────────────────────────────────────────────────────────

def cv_bar_chart(cv_results: dict) -> None:
    separator("10. CV COMPARISON BAR CHART")

    model_names  = list(cv_results.keys())
    bal_acc_vals = [cv_results[m]["CV Test Balanced Acc"] for m in model_names]
    bal_acc_stds = [cv_results[m]["CV Test Bal Acc Std"]  for m in model_names]
    f1_vals      = [cv_results[m]["CV Test Macro F1"]     for m in model_names]
    f1_stds      = [cv_results[m]["CV Test F1 Std"]       for m in model_names]

    x, width = np.arange(len(model_names)), 0.35
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, bal_acc_vals, width,
                   yerr=bal_acc_stds, capsize=5,
                   label="CV Balanced Accuracy", color="steelblue", alpha=0.85)
    bars2 = ax.bar(x + width/2, f1_vals, width,
                   yerr=f1_stds, capsize=5,
                   label="CV Macro F1", color="darkorange", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Cross-Validation Test Scores by Model", fontweight="bold")
    ax.legend(loc="lower right")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_and_show("cv_comparison_chart.png")


# ──────────────────────────────────────────────────────────────────────────────
# 11. CONFUSION MATRICES
# ──────────────────────────────────────────────────────────────────────────────

def confusion_matrices(svm_search, rf_search, lr_search, nb_search,
                        X_test_filt, y_test, class_names):
    separator("11. CONFUSION MATRICES")

    models = {
        "Kernel SVM":          svm_search.best_estimator_,
        "Random Forest":       rf_search.best_estimator_,
        "Logistic Regression": lr_search.best_estimator_,
        "Naive Bayes":         nb_search.best_estimator_,
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, (name, model) in zip(axes.flatten(), models.items()):
        y_pred = model.predict(X_test_filt)
        cm = confusion_matrix(y_test, y_pred).astype(float)
        cm_norm = cm / cm.sum(axis=1, keepdims=True)
        ConfusionMatrixDisplay(
            confusion_matrix=np.round(cm_norm, 2),
            display_labels=class_names,
        ).plot(ax=ax, colorbar=False, xticks_rotation=45)
        ax.set_title(name, fontweight="bold")

        # Print classification report
        print(f"\n  Classification Report — {name}")
        print(classification_report(y_test, y_pred,
                                    target_names=class_names, zero_division=0))

    plt.suptitle("Normalized Confusion Matrices (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_and_show("confusion_matrices.png")


# ──────────────────────────────────────────────────────────────────────────────
# 12. RANDOM FOREST FEATURE IMPORTANCES
# ──────────────────────────────────────────────────────────────────────────────

def feature_importances(rf_search, df, nonzero_mask):
    separator("12. RANDOM FOREST FEATURE IMPORTANCES")

    rf_best     = rf_search.best_estimator_.named_steps["clf"]
    importances = rf_best.feature_importances_
    top20_idx   = np.argsort(importances)[::-1][:20]

    feature_names  = np.array(df.drop(["samples", "type"], axis=1).columns)
    filtered_names = feature_names[nonzero_mask]
    top20_names    = filtered_names[top20_idx]
    top20_scores   = importances[top20_idx]

    print("\n  Top 20 Most Important Genes (Random Forest):")
    for rank, (gene, score) in enumerate(zip(top20_names, top20_scores), 1):
        print(f"  {rank:>2}. {gene:<20}  importance = {score:.6f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(np.arange(20), top20_scores[::-1], color="steelblue")
    ax.set_yticks(np.arange(20))
    ax.set_yticklabels(top20_names[::-1])
    ax.set_xlabel("Feature Importance")
    ax.set_title("Top 20 Most Important Genes (Random Forest)", fontweight="bold")
    plt.tight_layout()
    save_and_show("feature_importances.png")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cancer Gene Expression Classifier (Breast GSE45827)")
    parser.add_argument(
        "--data", default="Breast_GSE45827.csv",
        help="Path to the CSV dataset (default: Breast_GSE45827.csv)")
    args = parser.parse_args()

    if not os.path.isfile(args.data):
        raise FileNotFoundError(
            f"Dataset not found: {args.data}\n"
            "Download from: https://www.kaggle.com/datasets/brunogrisci/"
            "breast-cancer-gene-expression-cumida\n"
            "Then run: python main.py --data path/to/Breast_GSE45827.csv")

    # ── Pipeline ──────────────────────────────────────────────────────────────
    df = load_data(args.data)
    eda(df)

    (X_train_filt, X_test_filt,
     y_train, y_test,
     class_names, nonzero_mask, df) = preprocess(df)

    (svm_pipe, svm_grid,
     rf_pipe,  rf_grid,
     lr_pipe,  lr_grid,
     nb_pipe,  nb_grid) = build_pipelines_and_grids()

    svm_search, rf_search, lr_search, nb_search, cv = tune_models(
        X_train_filt, y_train,
        svm_pipe, svm_grid,
        rf_pipe,  rf_grid,
        lr_pipe,  lr_grid,
        nb_pipe,  nb_grid,
    )

    plot_learning_curves(svm_search, rf_search, lr_search, nb_search,
                          X_train_filt, y_train, cv)

    plot_validation_curves(svm_search, rf_search, lr_search, nb_search,
                            X_train_filt, y_train, cv)

    cv_results = cross_validation_evaluation(
        svm_search, rf_search, lr_search, nb_search,
        X_train_filt, y_train)

    results_table(svm_search, rf_search, lr_search, nb_search,
                   X_train_filt, X_test_filt, y_train, y_test)

    cv_bar_chart(cv_results)

    confusion_matrices(svm_search, rf_search, lr_search, nb_search,
                        X_test_filt, y_test, class_names)

    feature_importances(rf_search, df, nonzero_mask)

    separator("ALL DONE")
    print("  Figures saved in the current directory:")
    saved = [
        "eda_class_distribution.png",
        "lc_svm.png", "lc_rf.png", "lc_lr.png", "lc_nb.png",
        "vc_svm_C.png", "vc_lr_C.png", "vc_rf_nest.png", "vc_nb_vs.png",
        "results_table_final.png",
        "cv_comparison_chart.png",
        "confusion_matrices.png",
        "feature_importances.png",
    ]
    for f in saved:
        print(f"    • {f}")


if __name__ == "__main__":
    main()