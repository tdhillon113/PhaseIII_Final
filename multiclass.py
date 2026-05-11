

# ══════════════════════════════════════════════════════════════════════════════
#  DATASET PATH — edit this to point to your CSV file
# ══════════════════════════════════════════════════════════════════════════════
DATASET_PATH = "Breast_GSE45827.csv"
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import join, dirname, abspath, exists
from os import mkdir

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

# directory where plots are saved — matches the reference project layout
plots_directory = join(dirname(abspath(__file__)), "plots")
if not exists(plots_directory):
    mkdir(plots_directory)


def _save_fig(fig: plt.Figure, filename: str) -> None:
    """Save a specific figure to the plots directory and close it."""
    path = join(plots_directory, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [saved] {path}")


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
    fig_eda, ax = plt.subplots(figsize=(8, 5))
    df["type"].value_counts().plot(kind="bar", ax=ax, color="steelblue", edgecolor="white")
    ax.set_title("Class Distribution – Cancer Subtypes", fontweight="bold")
    ax.set_xlabel("Cancer Subtype")
    ax.set_ylabel("Sample Count")
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.show()
    _save_fig(fig_eda, "eda_class_distribution.png")

    # ── Summary statistics (numeric columns) ──────────────────────────────────
    print("\n  Summary Statistics (first 5 gene columns):")
    gene_cols = [c for c in df.columns if c not in ("samples", "type")]
    print(df[gene_cols[:5]].describe().round(4).to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING
# ──────────────────────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    separator("3. PREPROCESSING — Chain: AFFX Removal → Variance Filter → Train/Test Split")

    # Feature / label split
    X     = df.drop(["samples", "type"], axis=1).values.astype(np.float64)
    y_raw = df["type"].values

    # Encode labels
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    class_names = le.classes_
    print(f"  Classes ({len(class_names)}): {class_names}")
    print(f"  Class counts: {dict(zip(class_names, np.bincount(y)))}")

    # ── Step 1: Remove AFFX control probes ───────────────────────────────────
    feature_names = np.array(df.drop(["samples", "type"], axis=1).columns)
    affx_mask     = np.array([not name.startswith("AFFX") for name in feature_names])
    X_clean       = X[:, affx_mask]
    feature_names_clean = feature_names[affx_mask]

    print(f"\n  Step 1 — AFFX Probe Removal")
    print(f"  Original features  : {X.shape[1]:,}")
    print(f"  AFFX probes removed: {np.sum(~affx_mask):,}")
    print(f"  Remaining features : {X_clean.shape[1]:,}")

    # ── Step 2: Train/Test Split on cleaned data ──────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X_clean, y, test_size=0.3, random_state=42, stratify=y
    )
    X_train = np.asarray(X_train, dtype=np.float64)
    X_test  = np.asarray(X_test,  dtype=np.float64)
    y_train = np.asarray(y_train)
    y_test  = np.asarray(y_test)

    # ── Step 3: Variance Threshold Filtering (std > 0.05) ────────────────────
    stds     = np.std(X_train, axis=0)
    var_mask = stds > 0.05
    X_train_filt       = X_train[:, var_mask]
    X_test_filt        = X_test[:,  var_mask]
    feature_names_filt = feature_names_clean[var_mask]

    print(f"\n  Step 2 — Variance Filtering (std > 0.05)")
    print(f"  Features before : {X_train.shape[1]:,}")
    print(f"  Features removed: {np.sum(~var_mask):,}")
    print(f"  Features after  : {X_train_filt.shape[1]:,}")
    print(f"\n  Train size : {X_train_filt.shape[0]}")
    print(f"  Test size  : {X_test_filt.shape[0]}")

    return X_train_filt, X_test_filt, y_train, y_test, class_names, feature_names_filt, df


# ──────────────────────────────────────────────────────────────────────────────
# 4. MODEL DEFINITIONS
# ──────────────────────────────────────────────────────────────────────────────

def build_pipelines_and_grids():
    # Kernel SVM — StandardScaler + PCA (95% variance)
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

    # Random Forest — no SelectKBest; RF does internal feature selection
    rf_pipe = Pipeline([
        ("clf", RandomForestClassifier(
            class_weight="balanced", n_jobs=-1, random_state=42)),
    ])
    rf_grid = {
        "clf__n_estimators": [100, 200, 300],
        "clf__max_features": ["sqrt", "log2"],
    }

    # Logistic Regression — SelectKBest inside pipeline (no data leakage)
    # K range 100–500: aggressive selection on already-filtered feature space
    lr_pipe = Pipeline([
        ("select", SelectKBest(f_classif, k=200)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=10000, tol=0.01,
            class_weight="balanced", solver="saga", random_state=42)),
    ])
    lr_grid = {
        "select__k":    [100, 200, 300, 500],
        "clf__C":       [0.01, 0.1, 1, 10],
        "clf__penalty": ["l1", "l2"],
    }

    # Naive Bayes — SelectKBest inside pipeline (no data leakage)
    nb_pipe = Pipeline([
        ("select", SelectKBest(f_classif, k=200)),
        ("clf",    GaussianNB()),
    ])
    nb_grid = {
        "select__k":         [100, 200, 300, 500],
        "clf__var_smoothing": np.logspace(-9, -3, 4),
    }

    # Print pipeline summary
    print("\n" + "=" * 55)
    print("  PIPELINE SUMMARY")
    print("=" * 55)
    for name, pipe in [("Kernel SVM", svm_pipe), ("Random Forest", rf_pipe),
                        ("Logistic Regression", lr_pipe), ("Naive Bayes", nb_pipe)]:
        steps = " → ".join([s[0] for s in pipe.steps])
        print(f"  {name:<22}: {steps}")
    print("=" * 55)

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

    lc_figs = []
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
        lc_figs.append((fig, filename))

    plt.show()
    for fig, filename in lc_figs:
        _save_fig(fig, filename)


# ──────────────────────────────────────────────────────────────────────────────
# 7. VALIDATION CURVES
# ──────────────────────────────────────────────────────────────────────────────

def plot_validation_curves(svm_search, rf_search, lr_search, nb_search,
                            X_train_filt, y_train, cv):
    separator("7. VALIDATION CURVES")

    scoring = "balanced_accuracy"

    vc_figs = []

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
        vc_figs.append((fig, filename))

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

    plt.show()
    for fig, filename in vc_figs:
        _save_fig(fig, filename)


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
def styled_results_table(svm_search, rf_search, lr_search, nb_search,
                          X_test_filt, y_test, class_names, cv_results):
    separator("9b. STYLED SUMMARY RESULTS TABLE")

    models = {
        "Kernel SVM":          svm_search.best_estimator_,
        "Random Forest":       rf_search.best_estimator_,
        "Logistic Regression": lr_search.best_estimator_,
        "Naive Bayes":         nb_search.best_estimator_,
    }
    hyperparams = {
        "Kernel SVM":          f"kernel: rbf\nC: {svm_search.best_params_['clf__C']}\ngamma: {svm_search.best_params_['clf__gamma']}",
        "Random Forest":       f"n_estimators: {rf_search.best_params_['clf__n_estimators']}\nmax_features: {rf_search.best_params_['clf__max_features']}",
        "Logistic Regression": f"C: {lr_search.best_params_['clf__C']}\npenalty: {lr_search.best_params_['clf__penalty']}\nSelectKBest k={lr_search.best_params_['select__k']}",
        "Naive Bayes":         f"var_smoothing: {nb_search.best_params_['clf__var_smoothing']:.0e}\nSelectKBest k={nb_search.best_params_['select__k']}",
    }

    rows = []
    for name, model in models.items():
        y_pred        = model.predict(X_test_filt)
        test_bal_acc  = balanced_accuracy_score(y_test, y_pred)
        test_macro_f1 = f1_score(y_test, y_pred, average="macro")
        train_bal_acc = cv_results[name]["Train Balanced Acc"]
        train_macro_f1 = cv_results[name]["Train Macro F1"]
        cv_bal_acc    = cv_results[name]["CV Test Balanced Acc"]
        cv_macro_f1   = cv_results[name]["CV Test Macro F1"]
        rows.append({
            "model":          name,
            "params":         hyperparams[name],
            "train_bal_acc":  f"{train_bal_acc:.4f}",
            "train_macro_f1": f"{train_macro_f1:.4f}",
            "cv_bal_acc":     f"{cv_bal_acc:.4f}",
            "cv_macro_f1":    f"{cv_macro_f1:.4f}",
            "test_bal_acc":   f"{test_bal_acc:.4f}",
            "test_macro_f1":  f"{test_macro_f1:.4f}",
        })
        print(f"  {name}:")
        print(f"    Train  — Bal Acc: {train_bal_acc:.4f}  Macro F1: {train_macro_f1:.4f}")
        print(f"    CV     — Bal Acc: {cv_bal_acc:.4f}  Macro F1: {cv_macro_f1:.4f}")
        print(f"    Test   — Bal Acc: {test_bal_acc:.4f}  Macro F1: {test_macro_f1:.4f}")

    # ── Build figure ──────────────────────────────────────────────────────────
    col_headers = [
        "Model",
        "Hyperparameters",
        "Train\nBal Acc",
        "Train\nMacro F1",
        "CV\nBal Acc",
        "CV\nMacro F1",
        "Test\nBal Acc",
        "Test\nMacro F1",
    ]
    n_rows = len(rows)

    fig_h = 1.0 + n_rows * 1.6
    fig_w = 24

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")

    col_widths = [3.2, 4.8, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # must sum to fig_w (20 content + 4 for model/params)
    col_x = [0]
    for w in col_widths[:-1]:
        col_x.append(col_x[-1] + w)

    header_h   = 0.9
    data_row_h = (fig_h - header_h) / n_rows
    header_color  = "#4a4a6a"
    group_colors  = {
        "train": "#d4e6f1",   # light blue tint for training columns
        "cv":    "#d5f5e3",   # light green tint for CV columns
        "test":  "#fdebd0",   # light orange tint for test columns
    }
    row_colors = ["#ffffff", "#f0f0f0"]

    def draw_cell(x, y, w, h, text, bg, fg="black", bold=False, fontsize=10):
        rect = plt.Rectangle((x, y), w, h,
                              facecolor=bg, edgecolor="#cccccc", linewidth=0.6)
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text,
                ha="center", va="center", fontsize=fontsize,
                color=fg, fontweight="bold" if bold else "normal",
                multialignment="center", linespacing=1.4)

    # Draw column group labels above headers
    group_label_h = 0.45
    group_specs = [
        ("Training",   2, group_colors["train"]),
        ("Cross-Val",  2, group_colors["cv"]),
        ("Test",       2, group_colors["test"]),
    ]
    group_start_x = col_x[2]  # groups start after Model + Hyperparams columns
    gx = group_start_x
    for label, span, color in group_specs:
        gw = sum(col_widths[2 + group_specs.index((label, span, color)) * 2:
                            2 + group_specs.index((label, span, color)) * 2 + span])
        draw_cell(gx, fig_h - group_label_h, gw, group_label_h,
                  label, bg=color, fg="#2c2c2c", bold=True, fontsize=10)
        gx += gw

    # Draw column headers
    y_top = fig_h - group_label_h - header_h
    col_bg = (
        ["#4a4a6a", "#4a4a6a"]                          # model, hyperparams
        + [group_colors["train"]] * 2                   # train cols
        + [group_colors["cv"]] * 2                      # cv cols
        + [group_colors["test"]] * 2                    # test cols
    )
    col_fg = ["white", "white"] + ["#2c2c2c"] * 6

    for hdr, w, x, bg, fg in zip(col_headers, col_widths, col_x, col_bg, col_fg):
        draw_cell(x, y_top, w, header_h, hdr,
                  bg=bg, fg=fg, bold=True, fontsize=9.5)

    # Draw data rows
    col_group = ["", "", "train", "train", "cv", "cv", "test", "test"]
    for i, row in enumerate(rows):
        base_bg = row_colors[i % 2]
        y = y_top - (i + 1) * data_row_h
        cells = [
            row["model"],
            row["params"],
            row["train_bal_acc"],
            row["train_macro_f1"],
            row["cv_bal_acc"],
            row["cv_macro_f1"],
            row["test_bal_acc"],
            row["test_macro_f1"],
        ]
        for j, (cell_text, w, x) in enumerate(zip(cells, col_widths, col_x)):
            # Tint metric columns with their group color, blended with row stripe
            if col_group[j] == "train":
                bg = "#eaf4fb" if i % 2 == 0 else "#d4e6f1"
            elif col_group[j] == "cv":
                bg = "#eafaf1" if i % 2 == 0 else "#d5f5e3"
            elif col_group[j] == "test":
                bg = "#fef9f0" if i % 2 == 0 else "#fdebd0"
            else:
                bg = base_bg
            draw_cell(x, y, w, data_row_h, cell_text, bg=bg, bold=(j == 0))

    plt.title("Breast Cancer Gene Expression — Model Results Summary\n"
              "Balanced Accuracy  |  Macro F1  ·  across Training, Cross-Validation, and Test",
              fontsize=12, fontweight="bold", pad=14)
    plt.tight_layout()
    plt.figure(fig.number)
    plt.show(block=True)
    _save_fig(fig, "styled_results_table.png")

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
    ax.set_title("Cross-Validation Performance Scores by Model", fontweight="bold")
    ax.legend(loc="lower right")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{bar.get_height():.3f}",
                ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    plt.show()
    _save_fig(fig, "cv_comparison_chart.png")

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

        print(f"\n  Classification Report — {name}")
        print(classification_report(y_test, y_pred,
                                    target_names=class_names, zero_division=0))

    plt.suptitle("Normalized Confusion Matrices (Test Set)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    _save_fig(fig, "confusion_matrices.png")


# ──────────────────────────────────────────────────────────────────────────────
# 12. RANDOM FOREST FEATURE IMPORTANCES
# ──────────────────────────────────────────────────────────────────────────────

def feature_importances(rf_search, feature_names_filt):
    separator("12. RANDOM FOREST FEATURE IMPORTANCES")

    rf_best     = rf_search.best_estimator_.named_steps["clf"]
    importances = rf_best.feature_importances_
    top20_idx   = np.argsort(importances)[::-1][:20]

    top20_names  = feature_names_filt[top20_idx]
    top20_scores = importances[top20_idx]

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
    plt.show()
    _save_fig(fig, "feature_importances.png")



# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cancer Gene Expression Classifier (Breast GSE45827)")
    parser.add_argument(
        "--data", default=None,
        help="Path to the CSV dataset (overrides DATASET_PATH at top of file)")
    args = parser.parse_args()

    csv_path = args.data if args.data else DATASET_PATH

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"\n  Dataset not found: '{csv_path}'\n\n"
            "  Fix option 1 — edit DATASET_PATH near the top of main.py:\n"
            "      DATASET_PATH = \"path/to/Breast_GSE45827.csv\"\n\n"
            "  Fix option 2 — pass the path on the command line:\n"
            "      python main.py --data path/to/Breast_GSE45827.csv\n\n"
            "  Download the dataset from Kaggle:\n"
            "      https://www.kaggle.com/datasets/brunogrisci/"
            "breast-cancer-gene-expression-cumida")

    df = load_data(csv_path)
    eda(df)

    (X_train_filt, X_test_filt,
     y_train, y_test,
     class_names, feature_names_filt, df) = preprocess(df)

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

    adjusted_cv_evaluation(svm_search, rf_search, lr_search, nb_search,
                            X_train_filt, y_train, cv_results)

    styled_results_table(svm_search, rf_search, lr_search, nb_search,
                          X_test_filt, y_test, class_names, cv_results)

    cv_bar_chart(cv_results)

    confusion_matrices(svm_search, rf_search, lr_search, nb_search,
                        X_test_filt, y_test, class_names)

    feature_importances(rf_search, feature_names_filt)

    separator("ALL DONE")
    print(f"  All figures saved to: {plots_directory}/")
    saved = [
        "eda_class_distribution.png",
        "lc_svm.png", "lc_rf.png", "lc_lr.png", "lc_nb.png",
        "vc_svm_C.png", "vc_lr_C.png", "vc_rf_nest.png", "vc_nb_vs.png",
        "kfold_comparison_k5_vs_k7.png",
        "styled_results_table.png",
        "cv_comparison_chart.png",
        "confusion_matrices.png",
        "feature_importances.png",
    ]
    for fname in saved:
        print(f"    • {fname}")


if __name__ == "__main__":
    main()
