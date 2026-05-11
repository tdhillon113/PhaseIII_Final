from pathlib import Path

from binary_class_utils import (
    clean_wine_quality_data,
    generate_all_graphs,
    prepare_binary_data,
    run_all_binary_models,
    run_logistic_regression,
    run_random_forest,
    save_confusion_matrix,
    plot_learning_curve,
    plot_logistic_tuning_curve,
    plot_svm_c_tuning_curve,
    plot_svm_gamma_tuning_curve,
    plot_decision_tree_tuning_curve,
    plot_random_forest_tuning_curve,
)


def main():
    csv_path = "wine_quality_merged.csv"
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)

    print("Cleaning dataset...")
    df = clean_wine_quality_data(csv_path, save_cleaned=True)

    # PHASE 2: basic EDA
    print("Generating EDA plots...")
    generate_all_graphs(df, plots_dir)

    print("Running all binary classification models...")
    results_df, trained_models, confusion_matrices, best_params = run_all_binary_models(csv_path)

    print("Model comparison results:")
    summary_df = results_df[
        [
            "model",
            "train_accuracy",
            "val_accuracy",
            "test_accuracy",
            "train_f1_high_quality",
            "val_f1_high_quality",
            "test_f1_high_quality",
            "train_roc_auc",
            "val_roc_auc",
            "test_roc_auc",
        ]
    ].copy()

    summary_df = summary_df.rename(
        columns={
            "train_accuracy": "train_acc",
            "val_accuracy": "val_acc",
            "test_accuracy": "test_acc",
            "train_f1_high_quality": "train_F1",
            "val_f1_high_quality": "val_F1",
            "test_f1_high_quality": "test_F1",
            "train_roc_auc": "train_AUC",
            "val_roc_auc": "val_AUC",
            "test_roc_auc": "test_AUC",
        }
    )

    summary_df = summary_df.round(3)

    print(summary_df.to_string(index=False))

    print("\nBest hyperparameters:")
    for model_name, params in best_params.items():
        print(f"{model_name}: {params}")

    # Confusion matrices
    for model_name, cm in confusion_matrices.items():
        filename = model_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        save_confusion_matrix(cm, model_name, plots_dir / f"{filename}_confusion_matrix.png")

    X_train, X_test, y_train, y_test = prepare_binary_data(csv_path)

    # PHASE 2: tuning curves
    print("Generating tuning curves...")
    plot_logistic_tuning_curve(
        X_train,
        y_train,
        plots_dir / "logistic_regression_tuning_curve.png"
    )
    plot_svm_c_tuning_curve(
        X_train,
        y_train,
        plots_dir / "svm_tuning_curve_c.png"
    )
    plot_svm_gamma_tuning_curve(
        X_train,
        y_train,
        plots_dir / "svm_tuning_curve_gamma.png"
    )
    plot_decision_tree_tuning_curve(
        X_train,
        y_train,
        plots_dir / "decision_tree_tuning_curve.png"
    )
    plot_random_forest_tuning_curve(
        X_train,
        y_train,
        plots_dir / "random_forest_tuning_curve.png"
    )
    print("Saved tuning curves.")

    # PHASE 2: learning curves for all 4 models
    print("Generating learning curves...")
    lr_model = trained_models["Logistic Regression"]
    svm_model = trained_models["SVM (RBF)"]
    dt_model = trained_models["Decision Tree"]
    rf_model = trained_models["Random Forest"]

    plot_learning_curve(
        lr_model,
        X_train,
        y_train,
        "Logistic Regression Learning Curve",
        plots_dir / "logistic_regression_learning_curve.png",
    )
    plot_learning_curve(
        svm_model,
        X_train,
        y_train,
        "SVM (RBF) Learning Curve",
        plots_dir / "svm_learning_curve.png",
    )
    plot_learning_curve(
        dt_model,
        X_train,
        y_train,
        "Decision Tree Learning Curve",
        plots_dir / "decision_tree_learning_curve.png",
    )
    plot_learning_curve(
        rf_model,
        X_train,
        y_train,
        "Random Forest Learning Curve",
        plots_dir / "random_forest_learning_curve.png",
    )
    print("Done.")


if __name__ == "__main__":
    main()