# Binary Classification of Wine Quality and Multiclass Classification of Breast Cancer Subtype

## This project implements machine learning pipelines for two classification tasks:

```
1. Binary Classification: Predicting wine quality (high vs. low) using physicochemical properties
2. Multiclass Classification: Classifying breast cancer subtypes using gene expression data
```
The project includes data preprocessing, exploratory data analysis (EDA), model training with hyperparameter tuning, cross-validation, and comprehensive evaluation metrics.

## Features

> - Data Preprocessing: Cleaning, feature engineering, and handling class imbalance
> - Exploratory Data Analysis: Statistical summaries and visualizations
> - Model Training: Multiple algorithms with hyperparameter optimization
> - Evaluation: Accuracy, F1-score, ROC-AUC, confusion matrices, learning curves
> - Visualization: Plots for model comparison, feature importance, and performance metrics
> - Automated Pipeline: End-to-end execution for both datasets


## Binary Classification: Wine Quality Prediction

| Property | Details |
|----------|---------|
| **Source** | UCI Machine Learning Repository |
| **Features** | 11 physicochemical properties (acidity, alcohol, etc.) |
| **Target** | Binary: quality ≥ 7 as "high", else "low" |
| **Samples** | ~6,500 instances |

| Model | Algorithm |
|-------|-----------|
| Logistic Regression | Linear classifier |
| Support Vector Machine | SVM with RBF kernel |
| Decision Tree | Tree-based classifier |
| Random Forest | Ensemble of decision trees |
| Naive Bayes | Probabilistic classifier |

##  Multiclass Classification: Breast Cancer Subtypes

| Property | Details |
|----------|---------|
| **Source** | Kaggle (Breast Cancer Gene Expression - CUMIDA) |
| **Features** | Gene expression levels |
| **Target** | Multiclass: cancer subtypes |
| **Samples** | Gene expression profiles |

| Model | Algorithm |
|-------|-----------|
| Logistic Regression | Multinomial logistic |
| Support Vector Machine | SVM with RBF kernel |
| Random Forest | Ensemble classifier |
| Naive Bayes | Gaussian NB |

**All models include:**
- Hyperparameter tuning (Grid/Randomized Search)
- Cross-validation (Stratified K-Fold)
- Feature selection/scaling where appropriate


##  Results and Outputs

### Generated Files
The `plots/` directory contains all visualizations:

| Plot Type | Description |
|-----------|-------------|
| Confusion Matrices | Model prediction accuracy |
| Learning Curves | Training vs validation performance |
| Validation Curves | Hyperparameter impact |
| Feature Importance | Key predictors |
| Model Comparison | Performance across algorithms |
| EDA Plots | Data distributions & correlations |


### Console Output
-  Model performance summaries
-  Best hyperparameters found
-  Cross-validation results

>  See `final_proj_env.yml` for complete environment specification.

## Installation

### Prerequisites
-  Conda or Miniconda installed
-  Python 3.13

### Setup Environment
1.  Clone or download the repository
2.  Navigate to the project directory
3.  Create the conda environment:
   ```bash
   conda env create -f final_proj_env.yml
   ```
4.  Activate the environment:
   ```bash
   conda activate final_proj_env
   ```

### Data Preparation
- Wine Quality Dataset : `wine_quality_merged.csv` should be in the root directory
- Breast Cancer Dataset : Download `Breast_GSE45827.csv` from [Kaggle](https://www.kaggle.com/datasets/brunogrisci/breast-cancer-gene-expression-cumida) and place it in the root directory

## To Run the complete pipeline:
```bash
python main.py
```

> **Note**: The binary classification section may take a significant amount of time due to hyperparameter tuning.


## Acknowledgments

- CSC 156 Professor Corey Elowsky for all his teachings!
- Wine Quality Dataset: UCI Machine Learning Repository
- Breast Cancer Dataset: Kaggle/CUMIDA
- Libraries: Scikit-learn, Pandas, Matplotlib, Seaborn
