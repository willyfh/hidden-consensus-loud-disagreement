import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
import json

# Load data
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget class distribution:")
print(df['class'].value_counts())

# Identify target and features
y = df['class'].copy()
X = df.drop('class', axis=1)

print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"Target encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Process features
X_processed = X.copy()

# Identify and encode categorical columns
categorical_cols = X_processed.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

print(f"\nProcessed features shape: {X_processed.shape}")
print(f"Data types after encoding:\n{X_processed.dtypes}")

# Handle any missing values (fill with 0 if any remain)
if X_processed.isnull().any().any():
    print("\nFilling missing values with 0")
    X_processed = X_processed.fillna(0)

print("\n" + "="*60)
print("MODEL EVALUATION WITH STRATIFIED 5-FOLD CV")
print("="*60)

# Set up stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Logistic Regression with scikit-learn defaults
print("\nTraining Logistic Regression (defaults)...")
lr_model = LogisticRegression()
lr_scores = cross_val_score(
    lr_model, X_processed, y_encoded,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1
)
lr_auc_mean = lr_scores.mean()
lr_auc_std = lr_scores.std()
print(f"LogisticRegression ROC-AUC scores: {lr_scores}")
print(f"Mean: {lr_auc_mean:.6f} (+/- {lr_auc_std:.6f})")

# Random Forest with scikit-learn defaults
print("\nTraining Random Forest (defaults)...")
rf_model = RandomForestClassifier()
rf_scores = cross_val_score(
    rf_model, X_processed, y_encoded,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1
)
rf_auc_mean = rf_scores.mean()
rf_auc_std = rf_scores.std()
print(f"RandomForest ROC-AUC scores: {rf_scores}")
print(f"Mean: {rf_auc_mean:.6f} (+/- {rf_auc_std:.6f})")

# Compare results
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
diff = rf_auc_mean - lr_auc_mean
print(f"\nRandom Forest ROC-AUC:  {rf_auc_mean:.6f}")
print(f"Logistic Regression ROC-AUC: {lr_auc_mean:.6f}")
print(f"Difference (RF - LogReg): {diff:.6f}")

if rf_auc_mean > lr_auc_mean:
    print("\n✓ RESULT: Random Forest achieves HIGHER ROC-AUC than Logistic Regression")
    direction = "RF > LogReg"
else:
    print("\n✗ RESULT: Random Forest does NOT achieve higher ROC-AUC than Logistic Regression")
    direction = "RF <= LogReg"

# Prepare results
results = {
    "hypothesis_id": "H2",
    "summary": f"Random Forest achieves a stratified 5-fold cross-validated ROC-AUC of {rf_auc_mean:.6f}, while Logistic Regression achieves {lr_auc_mean:.6f}. Random Forest {'outperforms' if rf_auc_mean > lr_auc_mean else 'does not outperform'} Logistic Regression by {abs(diff):.6f}.",
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(diff, 6),
    "direction": direction,
    "methodological_choices": "Used scikit-learn default hyperparameters for both models (RandomForestClassifier and LogisticRegression with no parameter overrides). Stratified 5-fold cross-validation with random_state=42 for reproducibility. Categorical features were label-encoded. Target variable (class: '<=50K' or '>50K') was label-encoded to binary format. No class weighting or resampling was applied. ROC-AUC metric used for evaluation as specified."
}

# Save results
with open('result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Results saved to result.json")
print("="*60)
print(json.dumps(results, indent=2))
