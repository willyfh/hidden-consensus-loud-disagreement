"""
Analysis of model calibration on Adult Income dataset.
Research Question: Is the model well-calibrated?
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('adult_income.csv')
print(f"Data shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Explore target variable
print(f"\nTarget variable distribution:")
print(df['class'].value_counts())
print(f"Target proportions:\n{df['class'].value_counts(normalize=True)}")

# Data preparation
print("\n" + "="*60)
print("DATA PREPARATION")
print("="*60)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target: >50K = 1, <=50K = 0
y = (y == '>50K').astype(int)
print(f"Target encoding: >50K=1, <=50K=0")
print(f"Positive class prevalence: {y.mean():.4f}")

# Handle categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Handle missing values
for col in X.columns:
    if X[col].isnull().sum() > 0:
        if col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
        else:
            X[col] = X[col].fillna(X[col].median())

# Encode categorical variables
le_dict = {}
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    le_dict[col] = le

print(f"Data shape after encoding: {X_encoded.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)
print(f"\nTrain set size: {X_train.shape[0]}, positive class: {y_train.mean():.4f}")
print(f"Test set size: {X_test.shape[0]}, positive class: {y_test.mean():.4f}")

# Train base model (Logistic Regression)
print("\n" + "="*60)
print("MODEL TRAINING AND CALIBRATION EVALUATION")
print("="*60)

print("\n1. Training Logistic Regression model...")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

# Get predictions on test set
y_pred_proba = lr.predict_proba(X_test)[:, 1]
y_pred = lr.predict(X_test)

# Calculate calibration metrics
print("\n2. Computing calibration metrics on test set...")

# Expected Calibration Error (ECE)
# Bin predictions and compare mean predicted probability to mean actual outcome
def calculate_ece(y_true, y_pred_proba, n_bins=10):
    """Calculate Expected Calibration Error"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            conf = y_pred_proba[mask].mean()
            acc = y_true[mask].mean()
            ece += (mask.sum() / len(y_true)) * abs(conf - acc)

    return ece

# Maximum Calibration Error (MCE)
def calculate_mce(y_true, y_pred_proba, n_bins=10):
    """Calculate Maximum Calibration Error"""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mce = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            conf = y_pred_proba[mask].mean()
            acc = y_true[mask].mean()
            mce = max(mce, abs(conf - acc))

    return mce

ece = calculate_ece(y_test.values, y_pred_proba, n_bins=10)
mce = calculate_mce(y_test.values, y_pred_proba, n_bins=10)
brier = brier_score_loss(y_test, y_pred_proba)
logloss = log_loss(y_test, y_pred_proba)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Expected Calibration Error (ECE): {ece:.4f}")
print(f"Maximum Calibration Error (MCE): {mce:.4f}")
print(f"Brier Score: {brier:.4f}")
print(f"Log Loss: {logloss:.4f}")
print(f"ROC-AUC: {roc_auc:.4f}")

# Calibration curve analysis
print("\n3. Analyzing calibration curve...")
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=10)
mean_pred_prob = (prob_true - prob_pred).mean()
print(f"Mean predicted probability: {y_pred_proba.mean():.4f}")
print(f"Actual positive rate in test set: {y_test.mean():.4f}")

# Assess calibration
print("\n4. Calibration Assessment:")
if ece < 0.05:
    calibration_status = "Well-calibrated (ECE < 0.05)"
elif ece < 0.10:
    calibration_status = "Reasonably calibrated (ECE < 0.10)"
elif ece < 0.15:
    calibration_status = "Moderately miscalibrated (ECE < 0.15)"
else:
    calibration_status = "Poorly calibrated (ECE >= 0.15)"

print(f"Status: {calibration_status}")

# Determine if over or under-confident
mean_confidence = y_pred_proba.mean()
actual_positive_rate = y_test.mean()
confidence_bias = mean_confidence - actual_positive_rate

if abs(confidence_bias) < 0.05:
    bias_status = "Minor bias"
elif confidence_bias > 0:
    bias_status = f"Over-confident (predicts {mean_confidence:.4f}, actual {actual_positive_rate:.4f})"
else:
    bias_status = f"Under-confident (predicts {mean_confidence:.4f}, actual {actual_positive_rate:.4f})"

print(f"{bias_status}")

# Cross-validation for stability check
print("\n" + "="*60)
print("STABILITY VALIDATION (5-Fold Cross-Validation)")
print("="*60)

cv_results = []
for seed in range(5):
    print(f"\nCV Fold with random seed {seed}...")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_encoded, y, test_size=0.3, random_state=seed, stratify=y
    )

    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(X_tr, y_tr)

    proba = model.predict_proba(X_te)[:, 1]

    fold_ece = calculate_ece(y_te.values, proba, n_bins=10)
    fold_mce = calculate_mce(y_te.values, proba, n_bins=10)
    fold_brier = brier_score_loss(y_te, proba)
    fold_logloss = log_loss(y_te, proba)

    cv_results.append({
        'seed': seed,
        'ece': fold_ece,
        'mce': fold_mce,
        'brier': fold_brier,
        'logloss': fold_logloss
    })

    print(f"  ECE: {fold_ece:.4f}, MCE: {fold_mce:.4f}, Brier: {fold_brier:.4f}, LogLoss: {fold_logloss:.4f}")

cv_df = pd.DataFrame(cv_results)
print(f"\nCross-validation Summary:")
print(f"ECE:     {cv_df['ece'].mean():.4f} ± {cv_df['ece'].std():.4f} (range: {cv_df['ece'].min():.4f} - {cv_df['ece'].max():.4f})")
print(f"MCE:     {cv_df['mce'].mean():.4f} ± {cv_df['mce'].std():.4f} (range: {cv_df['mce'].min():.4f} - {cv_df['mce'].max():.4f})")
print(f"Brier:   {cv_df['brier'].mean():.4f} ± {cv_df['brier'].std():.4f} (range: {cv_df['brier'].min():.4f} - {cv_df['brier'].max():.4f})")
print(f"LogLoss: {cv_df['logloss'].mean():.4f} ± {cv_df['logloss'].std():.4f} (range: {cv_df['logloss'].min():.4f} - {cv_df['logloss'].max():.4f})")

# Final summary
print("\n" + "="*60)
print("FINAL FINDINGS")
print("="*60)

is_well_calibrated = ece < 0.10

print(f"\nPrimary Finding:")
print(f"  Expected Calibration Error (ECE): {ece:.4f}")
print(f"  Model is {'well-calibrated' if is_well_calibrated else 'NOT well-calibrated'}")
print(f"\nStability Check:")
print(f"  ECE across 5 CV folds: {cv_df['ece'].mean():.4f} ± {cv_df['ece'].std():.4f}")
print(f"  Finding {'HOLDS UP' if cv_df['ece'].mean() < 0.10 else 'DOES NOT HOLD UP'} across different splits")

print("\n" + "="*60)
