from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_validate,
    learning_curve,
    train_test_split,
    validation_curve
)
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.pipeline import Pipeline
from imblearn.under_sampling import RandomUnderSampler

FEATURE_COLUMNS = [
    "fixed acidity",
    "volatile acidity",
    "citric acid",
    "residual sugar",
    "chlorides",
    "free sulfur dioxide",
    "total sulfur dioxide",
    "density",
    "pH",
    "sulphates",
    "alcohol",
    "type_bool",
]

TARGET_COLUMN = "quality_binary"
RANDOM_STATE = 11
N_SPLITS = 7

SCORING = {
    "accuracy": "accuracy",
    "precision_high_quality": "precision",
    "recall_high_quality": "recall",
    "f1_high_quality": "f1",
    "roc_auc": "roc_auc",
}

def clean_wine_quality_data(csv_path="wine_quality_merged.csv", save_cleaned=False, output_path=None):
    """
    Load and clean the wine dataset. 

    Steps performed:
    1. Read wine_quality_merged.csv
    2. Encode type: red -> 0, white -> 1 as type_bool
    3. Drop original 'type' column
    4. Convert quality into binary target:
       quality >= 7 -> 1 (high quality)
       quality < 7  -> 0 (low/average quality)
    5. Drop original 'quality' column
    6. Remove duplicate rows and reset index
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    df["type_bool"] = df["type"].map({"red": 0, "white": 1})
    df = df.drop(columns=["type"])

    # High-quality wines (7-9) -> 1, low/average wines (3-6) -> 0
    df[TARGET_COLUMN] = np.where(df["quality"] >= 7, 1, 0)
    df = df.drop(columns=["quality"])

    df = df.drop_duplicates().reset_index(drop=True)

    if save_cleaned:
        if output_path is None:
            output_path = csv_path.with_name("wine_quality_cleaned.csv")
        df.to_csv(output_path, index=False)

    return df

def prepare_binary_data(df, test_size=0.2):
    """
    Return train/test split (80/20) for the binary wine-quality task. :p
    """ 
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y, #stratified
    )

def _get_sampler():
    """
    Return a simple undersampler for the majority class.
    """
    return RandomUnderSampler(random_state=RANDOM_STATE)

def generate_all_graphs(df, output_dir="plots"):
    """
    Create basic EDA plots.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # target distribution
    sns.countplot(x=TARGET_COLUMN, data=df)
    plt.title("Distribution of Binary Wine Quality")
    plt.xlabel("quality_binary")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_dir / "quality_binary_distribution.png")
    plt.close()
    # alcohol vs quality
    sns.boxplot(x=TARGET_COLUMN, y="alcohol", data=df)
    plt.title("Alcohol by Binary Quality")
    plt.xlabel("quality_binary")
    plt.tight_layout()
    plt.savefig(output_dir / "alcohol_by_quality_boxplot.png")
    plt.close()
    # correlation heatmap
    numeric_df = df.select_dtypes(include=[np.number])
    sns.heatmap(numeric_df.corr(), annot=False, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png")
    plt.close()

def _get_cv():
    """
    Return a 7-fold stratified cross-validation splitter.
    """
    return StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

def _get_scores(model, X_test):
    """
    Return scores for ROC-AUC (probability for class 1).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_test)[:, 1]
    return model.decision_function(X_test)

def _evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """
    Fit a model, run 7-fold CV, and compute mean train/validation metrics and final test metrics.
    """
    cv = _get_cv() #how to split into folds
    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING, #my metrics
        return_train_score=True,
    )

    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    y_pred = fitted_model.predict(X_test)
    y_scores = _get_scores(fitted_model, X_test)

    test_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_high_quality": precision_score(y_test, y_pred),
        "recall_high_quality": recall_score(y_test, y_pred),
        "f1_high_quality": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_scores),
    }

    train_summary = {
        metric.replace("train_", "train_"): float(np.mean(scores))
        for metric, scores in cv_results.items()
        if metric.startswith("train_")
    }

    val_summary = {
        metric.replace("test_", "val_"): float(np.mean(scores))
        for metric, scores in cv_results.items()
        if metric.startswith("test_")
    }

    results = {
        "model": name,
        **train_summary,
        **val_summary,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    }

    cm = confusion_matrix(y_test, y_pred)
    return results, fitted_model, cm

def run_logistic_regression(X_train, y_train, X_test, y_test):
    """
    Train Logistic Regression with scaling, undersampling, and grid search.
    """
    model = Pipeline([
        ("sampler", _get_sampler()),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga"
        )),
    ])

    param_grid = {
        "model__C": [0.01, 0.1, 1, 10, 100], #regularization strength
        "model__l1_ratio": [0, 1], #L1 and L2 
    } #grid of hyperparameters to try 

    search = GridSearchCV(
        model,
        param_grid,
        scoring="f1", #optimize f1 on the positive class (high quality)
        cv=_get_cv(), #stratified k fold 10 splits
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    results, fitted_model, cm = _evaluate_model(
        "Logistic Regression",
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    return results, fitted_model, cm, best_params

def run_svm_rbf(X_train, y_train, X_test, y_test):
    """
    Train RBF-kernel SVM with scaling and grid search over C and gamma.
    No undersampling here.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            class_weight="balanced",
        )),
    ])

    param_grid = {
        "model__C": [0.1, 1, 10, 100],
        "model__gamma": ["scale", 0.01, 0.1, 1],
    }

    search = GridSearchCV(
        model,
        param_grid,
        scoring="f1",
        cv=_get_cv(),
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    results, fitted_model, cm = _evaluate_model(
        "SVM (RBF)",
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    return results, fitted_model, cm, best_params

def run_decision_tree(X_train, y_train, X_test, y_test):
    """
    Train Decision Tree with grid search over depth and leaf/split sizes.
    No undersampling here
    """
    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    param_grid = {
        "max_depth": [3, 5, 7, 10, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "min_samples_split": [2, 5, 10],
    }

    search = GridSearchCV(
        model,
        param_grid,
        scoring="f1",
        cv=_get_cv(),
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    results, fitted_model, cm = _evaluate_model(
        "Decision Tree",
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    return results, fitted_model, cm, best_params

def run_random_forest(X_train, y_train, X_test, y_test):
    """
    Train Random Forest with grid search over n_estimators, depth, and max_features.
    No undersampling here
    """
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "max_features": ["sqrt", "log2", None],
    }

    search = GridSearchCV(
        model,
        param_grid,
        scoring="f1",
        cv=_get_cv(),
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    best_params = search.best_params_

    results, fitted_model, cm = _evaluate_model(
        "Random Forest",
        best_model,
        X_train,
        y_train,
        X_test,
        y_test,
    )

    return results, fitted_model, cm, best_params

def run_all_binary_models(df):
    """
    Run all four models, and return a summary table and fitted models and confusion matrices.
    """
    X_train, X_test, y_train, y_test = prepare_binary_data(df)

    all_results = []
    trained_models = {}
    confusion_matrices = {}
    best_params = {}

    for run in [
        run_logistic_regression,
        run_svm_rbf,
        run_decision_tree,
        run_random_forest,
    ]:
        results, fitted_model, cm, params = run(X_train, y_train, X_test, y_test)
        all_results.append(results)
        trained_models[results["model"]] = fitted_model
        confusion_matrices[results["model"]] = cm
        best_params[results["model"]] = params

    results_df = pd.DataFrame(all_results)

    ordered_cols = [
        "model",
        "train_accuracy",
        "train_precision_high_quality",
        "train_recall_high_quality",
        "train_f1_high_quality",
        "train_roc_auc",
        "val_accuracy",
        "val_precision_high_quality",
        "val_recall_high_quality",
        "val_f1_high_quality",
        "val_roc_auc",
        "test_accuracy",
        "test_precision_high_quality",
        "test_recall_high_quality",
        "test_f1_high_quality",
        "test_roc_auc",
    ]

    results_df = results_df[ordered_cols].sort_values(
        by="test_f1_high_quality", ascending=False
    ).reset_index(drop=True)

    return results_df, trained_models, confusion_matrices, best_params

def plot_learning_curve(model, X, y, title, output_path):
    """
    Plot training and validation F1 vs number of training examples (10-fold CV).
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model,
        X,
        y,
        cv=_get_cv(),
        scoring="f1",
        train_sizes=np.linspace(0.1, 1.0, 10), #split into 10 parts
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    plt.figure()

    # shaded regions: mean ± std
    plt.fill_between(
        train_sizes,
        val_mean - val_std,
        val_mean + val_std,
        color="orange",
        alpha=0.3,
    )
    plt.fill_between(
        train_sizes,
        train_mean - train_std,
        train_mean + train_std,
        color="blue",
        alpha=0.2,
    )
    
    plt.plot(train_sizes, train_mean, marker="o", label="Training F1")
    plt.plot(train_sizes, val_mean, marker="s", label="Validation F1")
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("F1 score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

#tuning curves
def plot_tuning_curve(model, X, y, param_name, param_range, title, output_path):
    """
    Plot training and validation F1 with mean lines and mean±std shaded regions.
    """
    train_scores, val_scores = validation_curve(
        model,
        X,
        y,
        param_name=param_name,
        param_range=param_range,
        cv=_get_cv(),
        scoring="f1",
    )

    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)

    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    plt.figure()

    # shaded regions: mean ± std
    plt.fill_between(
        param_range,
        val_mean - val_std,
        val_mean + val_std,
        color="orange",
        alpha=0.3,
    )
    plt.fill_between(
        param_range,
        train_mean - train_std,
        train_mean + train_std,
        color="blue",
        alpha=0.2,
    )
    # mean lines
    plt.plot(param_range, train_mean, color="blue", marker="o", label="Training F1 (mean)")
    plt.plot(param_range, val_mean, color="orange", marker="s", label="Validation F1 (mean)")

    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("F1 score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_logistic_tuning_curve(X_train, y_train, output_path):
    """
    Plot Logistic Regression tuning curve for C.
    """
    model = Pipeline([
        ("sampler", _get_sampler()),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            solver="saga",
            l1_ratio=0
        )),
    ])

    plot_tuning_curve(
        model,
        X_train,
        y_train,
        "model__C",
        [0.01, 0.1, 1, 10, 100],
        "Logistic Regression Tuning Curve (C)",
        output_path
    )
def plot_svm_c_tuning_curve(X_train, y_train, output_path):
    """
    Plot SVM tuning curve for C.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            gamma="scale",
            class_weight="balanced"
        )),
    ])

    plot_tuning_curve(
        model,
        X_train,
        y_train,
        "model__C",
        [0.1, 1, 10, 100],
        "SVM Tuning Curve (C)",
        output_path,
    )

def plot_svm_gamma_tuning_curve(X_train, y_train, output_path):
    """
    Plot SVM tuning curve for gamma.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(
            kernel="rbf",
            C=1,
            class_weight="balanced"
        )),
    ])

    plot_tuning_curve(
        model,
        X_train,
        y_train,
        "model__gamma",
        [0.01, 0.1, 1],
        "SVM Tuning Curve (gamma)",
        output_path
    )

def plot_decision_tree_tuning_curve(X_train, y_train, output_path):
    """
    Plot Decision Tree tuning curve for max_depth.
    """
    model = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    plot_tuning_curve(
        model,
        X_train,
        y_train,
        "max_depth",
        [3, 5, 7, 10, 15, 20],
        "Decision Tree Tuning Curve (max_depth)",
        output_path,
    )

def plot_random_forest_tuning_curve(X_train, y_train, output_path):
    """
    Plot Random Forest tuning curve for n_estimators.
    """
    model = RandomForestClassifier(
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )

    plot_tuning_curve(
        model,
        X_train,
        y_train,
        "n_estimators",
        [50, 100, 200, 300],
        "Random Forest Tuning Curve (n_estimators)",
        output_path,
    )

def save_confusion_matrix(cm, title, output_path):
    """
    Save a confusion-matrix heatmap for a trained model.
    """
    plt.figure()
    sns.heatmap(cm) #annot=True if you want to see values 
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_test_score_comparison(trained_models, X_test, y_test, output_path):
    """
    Plot test Accuracy and F1 for each model.
    """
    model_order = ["Logistic Regression", "SVM (RBF)", "Decision Tree", "Random Forest"]

    acc_scores = []
    f1_scores = []

    for model_name in model_order:
        model = trained_models[model_name]

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        acc_scores.append(acc)
        f1_scores.append(f1)

    x = np.arange(len(model_order))
    width = 0.35

    plt.figure()
    plt.bar(x - width/2, acc_scores, width, label="Test Accuracy")
    plt.bar(x + width/2, f1_scores, width, label="Test F1")

    plt.title("Test Set Scores by Model")
    plt.ylabel("Score")
    plt.xticks(x, model_order, rotation=15)
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()