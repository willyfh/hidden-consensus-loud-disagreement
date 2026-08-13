"""
Analysis: Does RandomForest achieve higher ROC-AUC than LogisticRegression
on the Adult income dataset using stratified 5-fold cross-validation?
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import json

# Load the data
df = pd.read_csv('adult_income.csv')

# Encode the target variable
le_target = LabelEncoder()
y = le_target.fit_transform(df['class'])  # <=50K -> 0, >50K -> 1

# Separate features and target
X = df.drop('class', axis=1)

# Handle missing values in categorical columns
# For categorical features with missing values, fill with mode
categorical_cols = X.select_dtypes(include='object').columns
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')

# Encode categorical features using LabelEncoder
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Ensure all features are numeric
X = X.astype('float64')

print("Dataset prepared:")
print(f"  Shape: {X.shape}")
print(f"  Target distribution: {np.bincount(y)}")
print(f"  Features: {list(X.columns)}")

# Set up stratified 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize models with scikit-learn defaults
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(random_state=42, max_iter=1000)

# Compute ROC-AUC scores using stratified 5-fold cross-validation
print("\nRunning stratified 5-fold cross-validation...")

# RandomForest ROC-AUC scores
rf_scores = cross_val_score(
    rf, X, y, cv=skf, scoring='roc_auc', n_jobs=-1
)
print(f"RandomForest ROC-AUC fold scores: {rf_scores}")
rf_mean = rf_scores.mean()
rf_std = rf_scores.std()
print(f"RandomForest ROC-AUC: {rf_mean:.6f} (+/- {rf_std:.6f})")

# LogisticRegression ROC-AUC scores
lr_scores = cross_val_score(
    lr, X, y, cv=skf, scoring='roc_auc', n_jobs=-1
)
print(f"LogisticRegression ROC-AUC fold scores: {lr_scores}")
lr_mean = lr_scores.mean()
lr_std = lr_scores.std()
print(f"LogisticRegression ROC-AUC: {lr_mean:.6f} (+/- {lr_std:.6f})")

# Calculate the difference
diff = rf_mean - lr_mean
print(f"\nROC-AUC Difference (RF - LogReg): {diff:.6f}")

if diff > 0:
    direction = "RF > LogReg"
    summary = f"Random Forest achieves higher stratified 5-fold cross-validated ROC-AUC ({rf_mean:.4f}) than Logistic Regression ({lr_mean:.4f}) on the Adult income dataset, with a difference of {abs(diff):.4f}."
else:
    direction = "RF ≤ LogReg"
    summary = f"Logistic Regression achieves higher or equal stratified 5-fold cross-validated ROC-AUC ({lr_mean:.4f}) compared to Random Forest ({rf_mean:.4f}) on the Adult income dataset, with a difference of {abs(diff):.4f}."

# Prepare results
result = {
    "hypothesis_id": "H2",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": float(diff),
    "direction": direction,
    "methodological_choices": (
        "Missing value imputation: Categorical features with missing values were filled with the mode; "
        "Encoding: All categorical features were label-encoded; "
        "Validation scheme: Stratified 5-fold cross-validation with random_state=42; "
        "Model hyperparameters: RandomForestClassifier and LogisticRegression used scikit-learn defaults "
        "(RandomForest: n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42); "
        "(LogisticRegression: solver=lbfgs, C=1.0, max_iter=1000, random_state=42); "
        "Target encoding: <=50K mapped to 0, >50K mapped to 1; "
        "Metric: ROC-AUC (area under the receiver operating characteristic curve) for binary classification"
    )
}

# Print and save results
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"Random Forest ROC-AUC (mean): {rf_mean:.6f}")
print(f"Logistic Regression ROC-AUC (mean): {lr_mean:.6f}")
print(f"Difference (RF - LogReg): {diff:.6f}")
print(f"Direction: {direction}")
print("="*60)

# Save to JSON
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResults saved to result.json")
