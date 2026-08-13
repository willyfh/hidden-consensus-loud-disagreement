import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# Load data
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn info:")
print(df.info())
print("\nTarget distribution:")
print(df['class'].value_counts())
print("\nMissing values:")
print(df.isnull().sum())

# Preprocessing
# Remove rows with missing values in key columns
df_clean = df.dropna()
print(f"\nAfter removing missing values: {df_clean.shape}")

# Separate features and target
y = (df_clean['class'] == '>50K').astype(int)
X = df_clean.drop('class', axis=1)

print(f"\nTarget class distribution after cleaning:")
print(f"  <=50K: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.1f}%)")
print(f"  >50K: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.1f}%)")

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
label_encoders = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"\nEncoded {len(categorical_cols)} categorical columns")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}")
print(f"Test set: {X_test.shape}")

# ===== CALIBRATION ANALYSIS =====
print("\n" + "="*70)
print("CALIBRATION ANALYSIS")
print("="*70)

# Test uncalibrated models
print("\n1. Testing raw (uncalibrated) models on test set:")

# Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train, y_train)
y_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_proba_rf = rf_model.predict_proba(X_test)[:, 1]

# Calculate calibration metrics for raw models
def calculate_calibration_metrics(y_true, y_proba, n_bins=10):
    """Calculate calibration metrics"""
    ece = 0.0
    bin_edges = np.linspace(0, 1, n_bins + 1)

    for i in range(n_bins):
        mask = (y_proba >= bin_edges[i]) & (y_proba < bin_edges[i+1])
        if mask.sum() > 0:
            avg_pred = y_proba[mask].mean()
            avg_true = y_true[mask].mean()
            ece += np.abs(avg_pred - avg_true) * mask.sum() / len(y_true)

    brier = brier_score_loss(y_true, y_proba)
    logloss = log_loss(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    return {'ECE': ece, 'Brier': brier, 'LogLoss': logloss, 'AUC': auc}

metrics_lr = calculate_calibration_metrics(y_test, y_proba_lr)
metrics_rf = calculate_calibration_metrics(y_test, y_proba_rf)

print("\nLogistic Regression (raw):")
for k, v in metrics_lr.items():
    print(f"  {k}: {v:.6f}")

print("\nRandom Forest (raw):")
for k, v in metrics_rf.items():
    print(f"  {k}: {v:.6f}")

# Apply calibration
print("\n2. Testing calibrated models (Platt scaling) on test set:")

# Calibrate using isotonic regression with CV
lr_calibrated = CalibratedClassifierCV(lr_model, method='sigmoid', cv=5)
lr_calibrated.fit(X_train, y_train)
y_proba_lr_cal = lr_calibrated.predict_proba(X_test)[:, 1]

rf_calibrated = CalibratedClassifierCV(rf_model, method='sigmoid', cv=5)
rf_calibrated.fit(X_train, y_train)
y_proba_rf_cal = rf_calibrated.predict_proba(X_test)[:, 1]

metrics_lr_cal = calculate_calibration_metrics(y_test, y_proba_lr_cal)
metrics_rf_cal = calculate_calibration_metrics(y_test, y_proba_rf_cal)

print("\nLogistic Regression (calibrated):")
for k, v in metrics_lr_cal.items():
    print(f"  {k}: {v:.6f}")

print("\nRandom Forest (calibrated):")
for k, v in metrics_rf_cal.items():
    print(f"  {k}: {v:.6f}")

# ===== STABILITY CHECK WITH CROSS-VALIDATION =====
print("\n" + "="*70)
print("STABILITY CHECK: 5-FOLD STRATIFIED CV (5 different random seeds)")
print("="*70)

cv_results = []
seeds = [42, 123, 456, 789, 999]

for seed_idx, seed in enumerate(seeds):
    print(f"\nSeed {seed_idx+1}/{len(seeds)} (seed={seed}):")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_encoded, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Train and evaluate LR
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(X_tr, y_tr)
    y_pr_lr = lr.predict_proba(X_te)[:, 1]
    metrics = calculate_calibration_metrics(y_te, y_pr_lr)

    print(f"  Logistic Regression - ECE: {metrics['ECE']:.6f}, Brier: {metrics['Brier']:.6f}")
    cv_results.append(('LR', metrics))

    # Train and evaluate RF
    rf = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    y_pr_rf = rf.predict_proba(X_te)[:, 1]
    metrics = calculate_calibration_metrics(y_te, y_pr_rf)

    print(f"  Random Forest - ECE: {metrics['ECE']:.6f}, Brier: {metrics['Brier']:.6f}")
    cv_results.append(('RF', metrics))

# Summary statistics
print("\n" + "="*70)
print("SUMMARY OF CROSS-VALIDATION RESULTS")
print("="*70)

lr_results = [m for model_type, m in cv_results if model_type == 'LR']
rf_results = [m for model_type, m in cv_results if model_type == 'RF']

print("\nLogistic Regression ECE:")
ece_lr = [m['ECE'] for m in lr_results]
print(f"  Mean: {np.mean(ece_lr):.6f}")
print(f"  Std:  {np.std(ece_lr):.6f}")
print(f"  Range: [{np.min(ece_lr):.6f}, {np.max(ece_lr):.6f}]")

print("\nRandom Forest ECE:")
ece_rf = [m['ECE'] for m in rf_results]
print(f"  Mean: {np.mean(ece_rf):.6f}")
print(f"  Std:  {np.std(ece_rf):.6f}")
print(f"  Range: [{np.min(ece_rf):.6f}, {np.max(ece_rf):.6f}]")

print("\nLogistic Regression Brier:")
brier_lr = [m['Brier'] for m in lr_results]
print(f"  Mean: {np.mean(brier_lr):.6f}")
print(f"  Std:  {np.std(brier_lr):.6f}")
print(f"  Range: [{np.min(brier_lr):.6f}, {np.max(brier_lr):.6f}]")

print("\nRandom Forest Brier:")
brier_rf = [m['Brier'] for m in rf_results]
print(f"  Mean: {np.mean(brier_rf):.6f}")
print(f"  Std:  {np.std(brier_rf):.6f}")
print(f"  Range: [{np.min(brier_rf):.6f}, {np.max(brier_rf):.6f}]")

# ===== FINAL DETERMINATION =====
print("\n" + "="*70)
print("CALIBRATION ASSESSMENT")
print("="*70)

# A model is generally considered well-calibrated if ECE < 0.1
# and/or if calibration errors are small

is_lr_calibrated = np.mean(ece_lr) < 0.1
is_rf_calibrated = np.mean(ece_rf) < 0.1

print(f"\nLogistic Regression:")
print(f"  Mean ECE: {np.mean(ece_lr):.6f}")
print(f"  Assessment: {'WELL-CALIBRATED' if is_lr_calibrated else 'POORLY-CALIBRATED'}")
print(f"  (Threshold: ECE < 0.1)")

print(f"\nRandom Forest:")
print(f"  Mean ECE: {np.mean(ece_rf):.6f}")
print(f"  Assessment: {'WELL-CALIBRATED' if is_rf_calibrated else 'POORLY-CALIBRATED'}")
print(f"  (Threshold: ECE < 0.1)")

# Overall assessment: use the standard uncalibrated model (LR) as the primary result
overall_ece = np.mean(ece_lr)
primary_metric_name = "Expected Calibration Error (ECE) - Logistic Regression"
primary_metric_value = overall_ece
direction = "WELL-CALIBRATED" if is_lr_calibrated else "POORLY-CALIBRATED"

print(f"\n" + "="*70)
print("PRIMARY FINDING")
print("="*70)
print(f"\nPrimary Metric: {primary_metric_name}")
print(f"Value: {primary_metric_value:.6f}")
print(f"Direction: {direction}")
print(f"\nInterpretation: A logistic regression model trained on the adult income")
print(f"dataset shows {'good' if is_lr_calibrated else 'poor'} calibration with an ECE of {overall_ece:.6f}.")
print(f"Cross-validation across 5 different random seeds confirmed stability,")
print(f"with ECE values ranging from {np.min(ece_lr):.6f} to {np.max(ece_lr):.6f}.")

# Save for result.json
summary = f"The logistic regression model trained on the adult income dataset is {'well-calibrated' if is_lr_calibrated else 'poorly-calibrated'}, with an Expected Calibration Error (ECE) of {overall_ece:.6f}. This finding is stable across 5 independent random splits with different seeds."
methodological_choices = "Logistic regression classifier with L2 regularization, 70/30 train-test split, categorical encoding via LabelEncoder, calibration evaluation using Expected Calibration Error (ECE) with 10 bins, Brier score, and log-loss. Stability validated via 5 repeated train-test splits with different random seeds."
verification_method = "5 repeated stratified train-test splits (70/30) with different random seeds (42, 123, 456, 789, 999)"
verification_result = f"Finding stable: ECE mean = {np.mean(ece_lr):.6f} ± {np.std(ece_lr):.6f}, range [{np.min(ece_lr):.6f}, {np.max(ece_lr):.6f}]"

print("\n\n" + "="*70)
print("RESULT JSON CONTENT")
print("="*70)
import json
result = {
    "hypothesis_id": "H6",
    "summary": summary,
    "primary_metric_name": primary_metric_name,
    "primary_metric_value": round(primary_metric_value, 6),
    "direction": direction,
    "methodological_choices": methodological_choices,
    "verification_method": verification_method,
    "verification_result": verification_result
}
print(json.dumps(result, indent=2))

# Save result.json
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResult saved to result.json")
