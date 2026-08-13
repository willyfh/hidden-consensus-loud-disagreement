import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND PREPROCESS DATA
# ============================================================================

df = pd.read_csv('adult_income.csv')

# Encode target variable
df['target'] = (df['class'] == '>50K').astype(int)
X = df.drop(['class', 'target'], axis=1)
y = df['target']

# Handle missing values
# For categorical columns with missing values, fill with mode
# For workclass, occupation, native-country
X['workclass'] = X['workclass'].fillna(X['workclass'].mode()[0])
X['occupation'] = X['occupation'].fillna(X['occupation'].mode()[0])
X['native-country'] = X['native-country'].fillna(X['native-country'].mode()[0])

# Encode categorical variables
categorical_cols = X.select_dtypes(include='object').columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# ============================================================================
# 2. SPLIT DATA INTO TRAIN/CALIBRATION/TEST
# ============================================================================

# Use a three-way split to properly evaluate calibration
# Train set: fit the model
# Calibration set: fit calibration (if needed)
# Test set: evaluate calibration

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Reset indices for test and cal sets
y_test = y_test.reset_index(drop=True)
y_cal = y_cal.reset_index(drop=True)

print(f"Train set size: {len(X_train)}")
print(f"Calibration set size: {len(X_cal)}")
print(f"Test set size: {len(X_test)}")
print(f"Test set positive class frequency: {y_test.mean():.4f}")

# ============================================================================
# 3. TRAIN MODELS
# ============================================================================

# Logistic Regression (typically well-calibrated)
print("\n" + "="*70)
print("LOGISTIC REGRESSION")
print("="*70)
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr_train = lr.predict_proba(X_train)[:, 1]
y_pred_lr_cal = lr.predict_proba(X_cal)[:, 1]
y_pred_lr_test = lr.predict_proba(X_test)[:, 1]

# Random Forest (typically poorly calibrated due to class imbalance and bootstrap)
print("\nRANDOM FOREST")
print("="*70)
rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42,
                            n_jobs=-1, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf_train = rf.predict_proba(X_train)[:, 1]
y_pred_rf_cal = rf.predict_proba(X_cal)[:, 1]
y_pred_rf_test = rf.predict_proba(X_test)[:, 1]

# ============================================================================
# 4. EVALUATE CALIBRATION - LOGISTIC REGRESSION
# ============================================================================

print("\n" + "="*70)
print("CALIBRATION EVALUATION - LOGISTIC REGRESSION")
print("="*70)

# Brier Score (lower is better, 0 is perfect)
brier_lr = brier_score_loss(y_test, y_pred_lr_test)
print(f"Brier Score: {brier_lr:.6f}")

# Log Loss (cross-entropy)
logloss_lr = log_loss(y_test, y_pred_lr_test)
print(f"Log Loss: {logloss_lr:.6f}")

# Expected Calibration Error (ECE)
# Divide predictions into bins and compute calibration error
def compute_ece(y_true, y_pred, n_bins=10):
    """Compute Expected Calibration Error"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_sums = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)

    for i in range(len(y_true)):
        bin_idx = np.searchsorted(bin_edges[1:], y_pred[i])
        bin_total[bin_idx] += 1
        bin_sums[bin_idx] += y_true[i]

    ece = 0
    total_samples = len(y_true)

    for i in range(n_bins):
        if bin_total[i] > 0:
            accuracy = bin_sums[i] / bin_total[i]
            confidence = bin_centers[i]
            ece += (bin_total[i] / total_samples) * abs(accuracy - confidence)

    return ece

ece_lr = compute_ece(y_test, y_pred_lr_test, n_bins=10)
print(f"Expected Calibration Error (ECE): {ece_lr:.6f}")

# Maximum Calibration Error (MCE)
def compute_mce(y_true, y_pred, n_bins=10):
    """Compute Maximum Calibration Error"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_sums = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)

    for i in range(len(y_true)):
        bin_idx = np.searchsorted(bin_edges[1:], y_pred[i])
        bin_total[bin_idx] += 1
        bin_sums[bin_idx] += y_true[i]

    mce = 0

    for i in range(n_bins):
        if bin_total[i] > 0:
            accuracy = bin_sums[i] / bin_total[i]
            confidence = bin_centers[i]
            mce = max(mce, abs(accuracy - confidence))

    return mce

mce_lr = compute_mce(y_test, y_pred_lr_test, n_bins=10)
print(f"Maximum Calibration Error (MCE): {mce_lr:.6f}")

# Mean Calibration Error (MCE) - alternative definition
def compute_mean_calibration_error(y_true, y_pred, n_bins=10):
    """Compute mean absolute calibration error across bins"""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    errors = []

    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i+1])
        if mask.sum() > 0:
            accuracy = y_true[mask].mean()
            confidence = y_pred[mask].mean()
            errors.append(abs(accuracy - confidence))

    return np.mean(errors) if errors else 0

mce_alt_lr = compute_mean_calibration_error(y_test, y_pred_lr_test, n_bins=10)
print(f"Mean Absolute Calibration Error: {mce_alt_lr:.6f}")

# Absolute Calibration Error (overall)
ace_lr = np.abs(y_test.mean() - y_pred_lr_test.mean())
print(f"Absolute Calibration Error (mean prediction vs actual): {ace_lr:.6f}")

# ============================================================================
# 5. EVALUATE CALIBRATION - RANDOM FOREST
# ============================================================================

print("\n" + "="*70)
print("CALIBRATION EVALUATION - RANDOM FOREST")
print("="*70)

brier_rf = brier_score_loss(y_test, y_pred_rf_test)
print(f"Brier Score: {brier_rf:.6f}")

logloss_rf = log_loss(y_test, y_pred_rf_test)
print(f"Log Loss: {logloss_rf:.6f}")

ece_rf = compute_ece(y_test, y_pred_rf_test, n_bins=10)
print(f"Expected Calibration Error (ECE): {ece_rf:.6f}")

mce_rf = compute_mce(y_test, y_pred_rf_test, n_bins=10)
print(f"Maximum Calibration Error (MCE): {mce_rf:.6f}")

mce_alt_rf = compute_mean_calibration_error(y_test, y_pred_rf_test, n_bins=10)
print(f"Mean Absolute Calibration Error: {mce_alt_rf:.6f}")

ace_rf = np.abs(y_test.mean() - y_pred_rf_test.mean())
print(f"Absolute Calibration Error (mean prediction vs actual): {ace_rf:.6f}")

# ============================================================================
# 6. COMPARISON AND INTERPRETATION
# ============================================================================

print("\n" + "="*70)
print("CALIBRATION COMPARISON")
print("="*70)

print("\nLogistic Regression (Baseline - typically well-calibrated):")
print(f"  - Brier Score: {brier_lr:.6f}")
print(f"  - ECE: {ece_lr:.6f}")

print("\nRandom Forest (Typically less calibrated):")
print(f"  - Brier Score: {brier_rf:.6f} (Δ: {brier_rf - brier_lr:+.6f})")
print(f"  - ECE: {ece_rf:.6f} (Δ: {ece_rf - ece_lr:+.6f})")

print("\nInterpretation:")
if ece_lr < 0.05:
    print("✓ Logistic Regression is WELL-CALIBRATED (ECE < 0.05)")
else:
    print("✗ Logistic Regression shows MODERATE TO POOR CALIBRATION (ECE >= 0.05)")

if ece_rf > ece_lr:
    print(f"✓ Random Forest is LESS CALIBRATED than LR (ECE diff: {ece_rf - ece_lr:.6f})")
else:
    print(f"✗ Random Forest is MORE CALIBRATED than LR (unexpected)")

# Overall finding
print("\n" + "="*70)
print("OVERALL FINDING")
print("="*70)

# The primary metric is Expected Calibration Error for the default model (LR)
primary_metric_value = ece_lr
primary_metric_name = "Expected Calibration Error (ECE)"

if ece_lr < 0.02:
    summary = f"The logistic regression model is WELL-CALIBRATED with ECE of {ece_lr:.6f}, indicating excellent alignment between predicted probabilities and actual class frequencies."
    direction = "model is well-calibrated"
elif ece_lr < 0.05:
    summary = f"The logistic regression model shows GOOD CALIBRATION with ECE of {ece_lr:.6f}, though with some minor deviation between predicted probabilities and actual frequencies."
    direction = "model is reasonably well-calibrated"
else:
    summary = f"The logistic regression model shows POOR CALIBRATION with ECE of {ece_lr:.6f}, indicating notable misalignment between predicted probabilities and actual class frequencies."
    direction = "model is miscalibrated"

print(summary)

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================

import json

results = {
    "hypothesis_id": "H6",
    "summary": summary,
    "primary_metric_name": primary_metric_name,
    "primary_metric_value": float(primary_metric_value),
    "direction": direction,
    "methodological_choices": (
        "Logistic Regression model trained on 60% of data, evaluated on 20% test set. "
        "Calibration assessed using Expected Calibration Error (ECE) with 10 bins, "
        "Brier Score, and Log Loss. Data preprocessed by filling missing values with mode, "
        "encoding all categorical features with LabelEncoder, and using stratified train-test split. "
        "Compared against Random Forest (100 trees, max_depth=15) to contextualize calibration. "
        "Primary metric is ECE as it is the most comprehensive measure of calibration."
    )
}

with open('result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to result.json")
print("\nFinal JSON:")
print(json.dumps(results, indent=2))
