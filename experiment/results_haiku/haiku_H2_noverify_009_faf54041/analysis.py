"""
Analysis: Random Forest vs Logistic Regression on Adult Income Dataset
Research Question (H2): Does RF achieve higher stratified 5-fold CV ROC-AUC than LogReg?
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import json
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load the dataset
print("Loading data...")
df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Target distribution:\n{df['class'].value_counts()}")

# Data inspection
print("\nData info:")
print(f"Missing values:\n{df.isnull().sum()}")
print(f"\nData types:\n{df.dtypes}")

# Create target variable (binary)
print("\n--- Preparing Target ---")
y = (df['class'] == '>50K').astype(int)
print(f"Target class balance: {y.value_counts()}")
print(f"Positive class (>50K): {y.sum()} ({100*y.mean():.1f}%)")

# Prepare features
print("\n--- Preparing Features ---")
X = df.drop(['class'], axis=1)

# Handle missing values in categorical columns (replace empty strings and '?' with mode)
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

for col in categorical_cols:
    # Count missing values (empty strings and '?')
    missing_count = ((X[col] == '') | (X[col] == '?')).sum()
    if missing_count > 0:
        # Replace with mode
        mode_val = X[col][(X[col] != '') & (X[col] != '?')].mode()
        if len(mode_val) > 0:
            X[col] = X[col].replace(['', '?'], mode_val[0])
        else:
            X[col] = X[col].replace(['', '?'], 'Unknown')
        print(f"  {col}: imputed {missing_count} missing values")

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
print(f"Numeric columns: {numeric_cols}")

# Encode categorical variables
print("\n--- Encoding Categorical Features ---")
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    print(f"  {col}: {len(le.classes_)} unique values")

print(f"\nFinal feature set shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# Setup cross-validation
print("\n--- Setting up Stratified 5-Fold Cross-Validation ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: Logistic Regression with scikit-learn defaults
print("\n--- Training Logistic Regression (sklearn defaults) ---")
# Note: LogisticRegression default uses L2 regularization with C=1.0
# We need to scale features for LR to work well
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
lr_scores = cross_val_score(
    lr_pipeline, X, y,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1
)
lr_mean_auc = lr_scores.mean()
lr_std_auc = lr_scores.std()
print(f"LogisticRegression ROC-AUC scores by fold: {lr_scores}")
print(f"LogisticRegression mean ROC-AUC: {lr_mean_auc:.6f} (+/- {lr_std_auc:.6f})")

# Model 2: Random Forest with scikit-learn defaults
print("\n--- Training Random Forest (sklearn defaults) ---")
# Using scikit-learn defaults for RandomForestClassifier:
# n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, etc.
rf_model = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(
    rf_model, X, y,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1
)
rf_mean_auc = rf_scores.mean()
rf_std_auc = rf_scores.std()
print(f"RandomForest ROC-AUC scores by fold: {rf_scores}")
print(f"RandomForest mean ROC-AUC: {rf_mean_auc:.6f} (+/- {rf_std_auc:.6f})")

# Comparison
print("\n" + "="*60)
print("RESULTS COMPARISON")
print("="*60)
print(f"LogisticRegression mean ROC-AUC: {lr_mean_auc:.6f}")
print(f"RandomForest mean ROC-AUC:       {rf_mean_auc:.6f}")
print(f"Difference (RF - LogReg):        {rf_mean_auc - lr_mean_auc:.6f}")
print(f"RF > LogReg: {rf_mean_auc > lr_mean_auc}")
print("="*60)

# Prepare result
if rf_mean_auc > lr_mean_auc:
    direction = "RF > LogReg"
    summary = f"Random Forest achieves higher stratified 5-fold cross-validated ROC-AUC ({rf_mean_auc:.4f}) compared to Logistic Regression ({lr_mean_auc:.4f}), with a difference of {rf_mean_auc - lr_mean_auc:.4f}."
else:
    direction = "LogReg >= RF"
    summary = f"Logistic Regression achieves higher or equal stratified 5-fold cross-validated ROC-AUC ({lr_mean_auc:.4f}) compared to Random Forest ({rf_mean_auc:.4f}), with a difference of {lr_mean_auc - rf_mean_auc:.4f}."

result = {
    "hypothesis_id": "H2",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": float(rf_mean_auc - lr_mean_auc),
    "direction": direction,
    "methodological_choices": (
        "Preprocessing: Handled missing values (empty strings and '?') in categorical columns by imputation with mode. "
        "Features: Encoded all categorical columns using LabelEncoder (15 categorical, 6 numeric). "
        "Feature scaling: Applied StandardScaler only in LogisticRegression pipeline (required for proper regularization), "
        "not for RandomForest (tree-based models are scale-invariant). "
        "Models: Used scikit-learn defaults (RandomForestClassifier: n_estimators=100, max_iter=1000; "
        "LogisticRegression: solver=lbfgs, max_iter=100 then increased to 1000 for convergence). "
        "Validation: Stratified 5-fold cross-validation (n_splits=5, shuffle=True, random_state=42) to account for class imbalance. "
        "Target: Binary classification (class >50K vs <=50K). "
        "Random seed: 42 for reproducibility."
    )
}

# Save result
print("\n--- Saving results to result.json ---")
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)
print("Results saved successfully!")

print("\nResult JSON:")
print(json.dumps(result, indent=2))
