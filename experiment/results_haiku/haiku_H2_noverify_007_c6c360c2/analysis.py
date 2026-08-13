"""
H2: Random Forest vs Logistic Regression on Adult Income Dataset
Stratified 5-fold Cross-Validated ROC-AUC Comparison
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import json
import warnings

warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['class'].value_counts()}")

# Data preprocessing
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Handle missing values (represented as '?' in some columns)
for col in df.select_dtypes(include=['object']).columns:
    if df[col].astype(str).str.contains('\?').any():
        print(f"Found '?' in {col}, replacing with mode")
        mode_val = df[col][df[col] != '?'].mode()[0] if (df[col] != '?').any() else 'Unknown'
        df[col] = df[col].replace('?', mode_val)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"Target classes: {le_target.classes_}")
print(f"Encoded target values: {np.unique(y_encoded)}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Encode categorical features
X_encoded = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"Encoded {col}: {len(le.classes_)} unique values")

# Convert to numpy array
X_array = X_encoded.values
print(f"\nFinal feature matrix shape: {X_array.shape}")
print(f"Feature dtypes: {X_encoded.dtypes.unique()}")

# Model comparison using Stratified 5-fold cross-validation
print("\n" + "="*60)
print("MODEL COMPARISON: STRATIFIED 5-FOLD CV")
print("="*60)

# Set up stratified k-fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest Classifier (scikit-learn defaults)
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(random_state=42)  # Using defaults
rf_scores = cross_val_score(
    rf_model, X_array, y_encoded,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)
print(f"Random Forest ROC-AUC scores (per fold): {rf_scores}")
print(f"Random Forest Mean ROC-AUC: {rf_scores.mean():.6f} (+/- {rf_scores.std():.6f})")

# Logistic Regression (scikit-learn defaults)
print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)  # Increased max_iter for convergence
lr_scores = cross_val_score(
    lr_model, X_array, y_encoded,
    cv=cv,
    scoring='roc_auc',
    n_jobs=-1
)
print(f"Logistic Regression ROC-AUC scores (per fold): {lr_scores}")
print(f"Logistic Regression Mean ROC-AUC: {lr_scores.mean():.6f} (+/- {lr_scores.std():.6f})")

# Comparison
print("\n" + "="*60)
print("RESULTS")
print("="*60)

rf_mean = rf_scores.mean()
lr_mean = lr_scores.mean()
difference = rf_mean - lr_mean

print(f"\nRandom Forest Mean ROC-AUC:    {rf_mean:.6f}")
print(f"Logistic Regression Mean ROC-AUC: {lr_mean:.6f}")
print(f"Difference (RF - LogReg):      {difference:.6f}")

if rf_mean > lr_mean:
    direction = "RF > LogReg"
    print(f"\n✓ Random Forest OUTPERFORMS Logistic Regression")
else:
    direction = "RF <= LogReg"
    print(f"\n✗ Random Forest does NOT outperform Logistic Regression")

# Prepare result summary
summary = (
    f"Random Forest achieves a mean ROC-AUC of {rf_mean:.6f} while Logistic Regression "
    f"achieves {lr_mean:.6f} in stratified 5-fold cross-validation. "
    f"The difference is {abs(difference):.6f}, with "
    f"{'Random Forest performing better' if rf_mean > lr_mean else 'Logistic Regression performing better'}."
)

methodological_choices = (
    "Used LabelEncoder for categorical and target encoding; no scaling applied to preserve tree model interpretability. "
    "Handled missing values (represented as '?') by replacing with mode per column. "
    "Stratified 5-fold cross-validation ensures balanced class distribution across folds. "
    "Random Forest uses scikit-learn defaults (100 estimators, max_depth=None, min_samples_split=2, min_samples_leaf=1). "
    "Logistic Regression uses defaults with max_iter=1000 for convergence (standard default is 100, but increased for stability). "
    "ROC-AUC metric chosen as specified in research question. "
    "random_state=42 for reproducibility. "
    "No hyperparameter tuning applied—used scikit-learn out-of-box defaults as specified."
)

# Save results
results = {
    "hypothesis_id": "H2",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(difference, 6),
    "direction": direction,
    "methodological_choices": methodological_choices,
    "rf_mean_roc_auc": round(rf_mean, 6),
    "rf_std_roc_auc": round(rf_scores.std(), 6),
    "lr_mean_roc_auc": round(lr_mean, 6),
    "lr_std_roc_auc": round(lr_scores.std(), 6),
}

with open('result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Results saved to result.json")
print("="*60)
print(json.dumps(results, indent=2))
