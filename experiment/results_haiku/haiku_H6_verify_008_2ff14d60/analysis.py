import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')

# Basic exploration
print("Dataset shape:", df.shape)
print("\nClass distribution:")
print(df['class'].value_counts())
print("\nMissing values:")
print(df.isnull().sum())
print(df.isna().sum())

# Check for empty strings in categorical columns
print("\nEmpty string counts:")
for col in df.select_dtypes(include='object').columns:
    empty_count = (df[col] == '').sum()
    if empty_count > 0:
        print(f"{col}: {empty_count}")

# Data preprocessing
df_clean = df.copy()

# Replace empty strings with NaN, then handle missing values
for col in df_clean.select_dtypes(include='object').columns:
    df_clean[col] = df_clean[col].replace('', np.nan)

# Fill missing values with mode for categorical, median for numeric
for col in df_clean.columns:
    if df_clean[col].isnull().sum() > 0:
        if df_clean[col].dtype == 'object' or df_clean[col].dtype == 'string':
            mode_val = df_clean[col].mode()
            if len(mode_val) > 0:
                df_clean[col].fillna(mode_val[0], inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].median(), inplace=True)

print(f"\nMissing values after handling: {df_clean.isnull().sum().sum()}")

# Separate target
y = (df_clean['class'] == '>50K').astype(int)
X = df_clean.drop('class', axis=1)

print(f"\nTarget distribution: {(y.sum() / len(y) * 100):.2f}% positive class")

# Encode categorical variables
le_dict = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

print(f"\nFeatures shape after encoding: {X.shape}")

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

# ============================================================================
# MODEL 1: Logistic Regression (naturally well-calibrated)
# ============================================================================
print("\n" + "="*70)
print("MODEL 1: Logistic Regression")
print("="*70)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)

y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]
y_pred_lr = lr.predict(X_test)

from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss

acc_lr = accuracy_score(y_test, y_pred_lr)
auc_lr = roc_auc_score(y_test, y_pred_proba_lr)
brier_lr = brier_score_loss(y_test, y_pred_proba_lr)

print(f"Accuracy: {acc_lr:.4f}")
print(f"ROC-AUC: {auc_lr:.4f}")
print(f"Brier Score: {brier_lr:.4f}")

# Calculate Expected Calibration Error (ECE)
def calculate_ece(y_true, y_pred_proba, n_bins=10):
    """Calculate Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0
    for bin_id in range(n_bins):
        mask = bin_indices == bin_id
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred_proba[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)

    return ece

# Calculate Maximum Calibration Error (MCE)
def calculate_mce(y_true, y_pred_proba, n_bins=10):
    """Calculate Maximum Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    mce = 0
    for bin_id in range(n_bins):
        mask = bin_indices == bin_id
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred_proba[mask].mean()
            mce = max(mce, abs(bin_acc - bin_conf))

    return mce

ece_lr = calculate_ece(y_test.values, y_pred_proba_lr, n_bins=10)
mce_lr = calculate_mce(y_test.values, y_pred_proba_lr, n_bins=10)

print(f"Expected Calibration Error (ECE): {ece_lr:.4f}")
print(f"Maximum Calibration Error (MCE): {mce_lr:.4f}")

# Hosmer-Lemeshow test
from scipy.stats import chi2

def hosmer_lemeshow_test(y_true, y_pred_proba, n_bins=10):
    """Perform Hosmer-Lemeshow test."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    hl_stat = 0
    for bin_id in range(n_bins):
        mask = bin_indices == bin_id
        if mask.sum() > 1:
            observed_events = y_true[mask].sum()
            observed_non_events = (1 - y_true[mask]).sum()
            expected_events = y_pred_proba[mask].sum()
            expected_non_events = (1 - y_pred_proba[mask]).sum()

            if expected_events > 0 and expected_non_events > 0:
                hl_stat += (observed_events - expected_events)**2 / expected_events
                hl_stat += (observed_non_events - expected_non_events)**2 / expected_non_events

    p_value = 1 - chi2.cdf(hl_stat, df=n_bins-2)
    return hl_stat, p_value

hl_stat_lr, hl_pval_lr = hosmer_lemeshow_test(y_test.values, y_pred_proba_lr, n_bins=10)
print(f"Hosmer-Lemeshow test statistic: {hl_stat_lr:.4f}")
print(f"Hosmer-Lemeshow p-value: {hl_pval_lr:.4f}")

# ============================================================================
# MODEL 2: Random Forest (often poorly calibrated without calibration)
# ============================================================================
print("\n" + "="*70)
print("MODEL 2: Random Forest (without calibration)")
print("="*70)

rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]
y_pred_rf = rf.predict(X_test)

acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_pred_proba_rf)
brier_rf = brier_score_loss(y_test, y_pred_proba_rf)

print(f"Accuracy: {acc_rf:.4f}")
print(f"ROC-AUC: {auc_rf:.4f}")
print(f"Brier Score: {brier_rf:.4f}")

ece_rf = calculate_ece(y_test.values, y_pred_proba_rf, n_bins=10)
mce_rf = calculate_mce(y_test.values, y_pred_proba_rf, n_bins=10)

print(f"Expected Calibration Error (ECE): {ece_rf:.4f}")
print(f"Maximum Calibration Error (MCE): {mce_rf:.4f}")

hl_stat_rf, hl_pval_rf = hosmer_lemeshow_test(y_test.values, y_pred_proba_rf, n_bins=10)
print(f"Hosmer-Lemeshow test statistic: {hl_stat_rf:.4f}")
print(f"Hosmer-Lemeshow p-value: {hl_pval_rf:.4f}")

# ============================================================================
# VALIDATION: Cross-validation calibration assessment
# ============================================================================
print("\n" + "="*70)
print("VALIDATION: 5-Fold Cross-Validation Calibration Assessment")
print("="*70)

from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

ece_scores_lr = []
mce_scores_lr = []
ece_scores_rf = []
mce_scores_rf = []

fold = 1
for train_idx, val_idx in skf.split(X, y):
    X_train_cv, X_val_cv = X.iloc[train_idx], X.iloc[val_idx]
    y_train_cv, y_val_cv = y.iloc[train_idx], y.iloc[val_idx]

    # LR
    lr_cv = LogisticRegression(max_iter=1000, random_state=42)
    lr_cv.fit(X_train_cv, y_train_cv)
    y_pred_proba_lr_cv = lr_cv.predict_proba(X_val_cv)[:, 1]

    ece_lr_cv = calculate_ece(y_val_cv.values, y_pred_proba_lr_cv, n_bins=10)
    mce_lr_cv = calculate_mce(y_val_cv.values, y_pred_proba_lr_cv, n_bins=10)
    ece_scores_lr.append(ece_lr_cv)
    mce_scores_lr.append(mce_lr_cv)

    # RF
    rf_cv = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf_cv.fit(X_train_cv, y_train_cv)
    y_pred_proba_rf_cv = rf_cv.predict_proba(X_val_cv)[:, 1]

    ece_rf_cv = calculate_ece(y_val_cv.values, y_pred_proba_rf_cv, n_bins=10)
    mce_rf_cv = calculate_mce(y_val_cv.values, y_pred_proba_rf_cv, n_bins=10)
    ece_scores_rf.append(ece_rf_cv)
    mce_scores_rf.append(mce_rf_cv)

    print(f"Fold {fold}:")
    print(f"  LR - ECE: {ece_lr_cv:.4f}, MCE: {mce_lr_cv:.4f}")
    print(f"  RF - ECE: {ece_rf_cv:.4f}, MCE: {mce_rf_cv:.4f}")
    fold += 1

print(f"\nLogistic Regression - ECE mean ± std: {np.mean(ece_scores_lr):.4f} ± {np.std(ece_scores_lr):.4f}")
print(f"Logistic Regression - MCE mean ± std: {np.mean(mce_scores_lr):.4f} ± {np.std(mce_scores_lr):.4f}")
print(f"Random Forest - ECE mean ± std: {np.mean(ece_scores_rf):.4f} ± {np.std(ece_scores_rf):.4f}")
print(f"Random Forest - MCE mean ± std: {np.mean(mce_scores_rf):.4f} ± {np.std(mce_scores_rf):.4f}")

# ============================================================================
# ADDITIONAL VALIDATION: Different random seeds
# ============================================================================
print("\n" + "="*70)
print("VALIDATION: Repeated 80-20 splits with different random seeds")
print("="*70)

ece_scores_lr_repeated = []
ece_scores_rf_repeated = []

for seed in [42, 123, 456, 789, 999]:
    X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # LR
    lr_s = LogisticRegression(max_iter=1000, random_state=seed)
    lr_s.fit(X_train_s, y_train_s)
    y_pred_proba_lr_s = lr_s.predict_proba(X_test_s)[:, 1]
    ece_lr_s = calculate_ece(y_test_s.values, y_pred_proba_lr_s, n_bins=10)
    ece_scores_lr_repeated.append(ece_lr_s)

    # RF
    rf_s = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=seed, n_jobs=-1)
    rf_s.fit(X_train_s, y_train_s)
    y_pred_proba_rf_s = rf_s.predict_proba(X_test_s)[:, 1]
    ece_rf_s = calculate_ece(y_test_s.values, y_pred_proba_rf_s, n_bins=10)
    ece_scores_rf_repeated.append(ece_rf_s)

    print(f"Seed {seed}: LR ECE = {ece_lr_s:.4f}, RF ECE = {ece_rf_s:.4f}")

print(f"\nLogistic Regression - ECE range: [{min(ece_scores_lr_repeated):.4f}, {max(ece_scores_lr_repeated):.4f}]")
print(f"Logistic Regression - ECE mean: {np.mean(ece_scores_lr_repeated):.4f} ± {np.std(ece_scores_lr_repeated):.4f}")
print(f"Random Forest - ECE range: [{min(ece_scores_rf_repeated):.4f}, {max(ece_scores_rf_repeated):.4f}]")
print(f"Random Forest - ECE mean: {np.mean(ece_scores_rf_repeated):.4f} ± {np.std(ece_scores_rf_repeated):.4f}")

# ============================================================================
# SUMMARY AND INTERPRETATION
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: Model Calibration Assessment")
print("="*70)

print("\nLogistic Regression (test set):")
print(f"  - Expected Calibration Error: {ece_lr:.4f}")
print(f"  - Maximum Calibration Error: {mce_lr:.4f}")
print(f"  - Brier Score: {brier_lr:.4f}")
print(f"  - H-L p-value: {hl_pval_lr:.4f} (well-calibrated if p > 0.05)")

print("\nRandom Forest (test set):")
print(f"  - Expected Calibration Error: {ece_rf:.4f}")
print(f"  - Maximum Calibration Error: {mce_rf:.4f}")
print(f"  - Brier Score: {brier_rf:.4f}")
print(f"  - H-L p-value: {hl_pval_rf:.4f} (well-calibrated if p > 0.05)")

print("\nInterpretation:")
print("ECE and MCE closer to 0 indicates better calibration.")
print("Hosmer-Lemeshow p-value > 0.05 suggests good calibration.")
print(f"\nLogistic Regression is {'WELL-CALIBRATED' if ece_lr < 0.05 else 'POORLY-CALIBRATED'} (ECE={ece_lr:.4f})")
print(f"Random Forest is {'WELL-CALIBRATED' if ece_rf < 0.05 else 'POORLY-CALIBRATED'} (ECE={ece_rf:.4f})")

# Save summary for JSON report
summary_data = {
    'model_lr_ece': ece_lr,
    'model_rf_ece': ece_rf,
    'model_lr_mce': mce_lr,
    'model_rf_mce': mce_rf,
    'model_lr_brier': brier_lr,
    'model_rf_brier': brier_rf,
    'model_lr_hl_pval': hl_pval_lr,
    'model_rf_hl_pval': hl_pval_rf,
    'lr_ece_cv_mean': np.mean(ece_scores_lr),
    'lr_ece_cv_std': np.std(ece_scores_lr),
    'rf_ece_cv_mean': np.mean(ece_scores_rf),
    'rf_ece_cv_std': np.std(ece_scores_rf),
    'lr_ece_repeated_mean': np.mean(ece_scores_lr_repeated),
    'lr_ece_repeated_std': np.std(ece_scores_lr_repeated),
    'rf_ece_repeated_mean': np.mean(ece_scores_rf_repeated),
    'rf_ece_repeated_std': np.std(ece_scores_rf_repeated),
}

import json
with open('/tmp/calibration_summary.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("\nAnalysis complete. Summary saved.")
