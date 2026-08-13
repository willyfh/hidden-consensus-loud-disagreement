"""
Analysis: Impact of SMOTE oversampling on minority-class F1 score
Research question H5: Does SMOTE change minority-class F1 score by >0.02?
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import json
import warnings

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_SEED = 42

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: Load and explore data")
print("=" * 80)

df = pd.read_csv('adult_income.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nTarget distribution:")
print(df['class'].value_counts())

# ============================================================================
# 2. PREPROCESS DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: Preprocess data")
print("=" * 80)

# Make a copy for preprocessing
data = df.copy()

# Handle missing values (represented as empty strings or spaces in categorical columns)
categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('class')  # Target column

for col in categorical_cols:
    # Replace missing/spaces with mode or 'Unknown'
    data[col] = data[col].replace(['?', ' ?', ''], np.nan)
    if data[col].isnull().sum() > 0:
        # Fill with the most common value or 'Unknown'
        fill_value = data[col].mode()[0] if len(data[col].mode()) > 0 else 'Unknown'
        data[col].fillna(fill_value, inplace=True)

print(f"After handling missing values - remaining NaNs: {data.isnull().sum().sum()}")

# Encode categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(data['class'])  # 0: <=50K, 1: >50K
X = data.drop('class', axis=1)

print(f"\nAfter encoding:")
print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}")
print(f"Minority class (>50K) proportion: {y.sum() / len(y):.4f}")

# ============================================================================
# 3. TRAIN/TEST SPLIT
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: Train/Test Split")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
)

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# ============================================================================
# 4. BASELINE: NO RESAMPLING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: Baseline RF (No Resampling)")
print("=" * 80)

rf_baseline = RandomForestClassifier(random_state=RANDOM_SEED)
rf_baseline.fit(X_train, y_train)
y_pred_baseline = rf_baseline.predict(X_test)

f1_baseline = f1_score(y_test, y_pred_baseline, pos_label=1, zero_division=0)
print(f"\nBaseline RF - Minority class F1 score: {f1_baseline:.6f}")
print(f"\nClassification Report (Baseline):")
print(classification_report(y_test, y_pred_baseline, target_names=['<=50K', '>50K']))

# ============================================================================
# 5. WITH SMOTE
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: RF with SMOTE Oversampling")
print("=" * 80)

# Apply SMOTE to training data only
smote = SMOTE(random_state=RANDOM_SEED)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training data after SMOTE:")
print(f"  Shape: {X_train_smote.shape}")
print(f"  Class distribution: {np.bincount(y_train_smote)}")

# Train RF on SMOTE data
rf_smote = RandomForestClassifier(random_state=RANDOM_SEED)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)

f1_smote = f1_score(y_test, y_pred_smote, pos_label=1, zero_division=0)
print(f"\nSMOTE RF - Minority class F1 score: {f1_smote:.6f}")
print(f"\nClassification Report (SMOTE):")
print(classification_report(y_test, y_pred_smote, target_names=['<=50K', '>50K']))

# ============================================================================
# 6. PRIMARY FINDING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 6: Primary Finding")
print("=" * 80)

f1_difference = f1_smote - f1_baseline
threshold = 0.02

print(f"\nMinority-class F1 scores:")
print(f"  Baseline (no SMOTE): {f1_baseline:.6f}")
print(f"  With SMOTE:          {f1_smote:.6f}")
print(f"  Difference:          {f1_difference:.6f}")
print(f"  Threshold:           {threshold:.6f}")

if abs(f1_difference) > threshold:
    answer = "YES"
    direction = "difference > 0.02" if f1_difference > threshold else "difference < -0.02"
    print(f"\nAnswer to H5: {answer} - {direction}")
else:
    answer = "NO"
    print(f"\nAnswer to H5: {answer} - difference <= 0.02")

# ============================================================================
# 7. VALIDATION: REPEATED STRATIFIED K-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 7: Stability Validation (Repeated Stratified K-Fold CV)")
print("=" * 80)

n_splits = 5
n_repeats = 5
cv_results_baseline = []
cv_results_smote = []

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)

for repeat in range(n_repeats):
    print(f"\nRepeat {repeat + 1}/{n_repeats}:")
    repeat_results_baseline = []
    repeat_results_smote = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_train, y_train)):
        X_cv_train, X_cv_test = X_train.iloc[train_idx], X_train.iloc[test_idx]
        y_cv_train, y_cv_test = y_train[train_idx], y_train[test_idx]

        # Baseline
        rf_cv_base = RandomForestClassifier(random_state=RANDOM_SEED + repeat * 10 + fold)
        rf_cv_base.fit(X_cv_train, y_cv_train)
        y_cv_pred_base = rf_cv_base.predict(X_cv_test)
        f1_cv_base = f1_score(y_cv_test, y_cv_pred_base, pos_label=1, zero_division=0)
        repeat_results_baseline.append(f1_cv_base)

        # SMOTE
        X_cv_train_smote, y_cv_train_smote = smote.fit_resample(X_cv_train, y_cv_train)
        rf_cv_smote = RandomForestClassifier(random_state=RANDOM_SEED + repeat * 10 + fold)
        rf_cv_smote.fit(X_cv_train_smote, y_cv_train_smote)
        y_cv_pred_smote = rf_cv_smote.predict(X_cv_test)
        f1_cv_smote = f1_score(y_cv_test, y_cv_pred_smote, pos_label=1, zero_division=0)
        repeat_results_smote.append(f1_cv_smote)

    avg_baseline = np.mean(repeat_results_baseline)
    avg_smote = np.mean(repeat_results_smote)
    cv_results_baseline.extend(repeat_results_baseline)
    cv_results_smote.extend(repeat_results_smote)

    print(f"  Fold F1 scores (Baseline): {[f'{x:.4f}' for x in repeat_results_baseline]}")
    print(f"  Fold F1 scores (SMOTE):    {[f'{x:.4f}' for x in repeat_results_smote]}")
    print(f"  Mean F1 (Baseline): {avg_baseline:.6f}")
    print(f"  Mean F1 (SMOTE):    {avg_smote:.6f}")
    print(f"  Diff:               {avg_smote - avg_baseline:.6f}")

# ============================================================================
# 8. VALIDATION RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 8: Validation Summary")
print("=" * 80)

cv_results_baseline = np.array(cv_results_baseline)
cv_results_smote = np.array(cv_results_smote)
cv_differences = cv_results_smote - cv_results_baseline

print(f"\nCross-validation F1 scores (across all folds):")
print(f"  Baseline - Mean: {cv_results_baseline.mean():.6f}, Std: {cv_results_baseline.std():.6f}")
print(f"  SMOTE    - Mean: {cv_results_smote.mean():.6f}, Std: {cv_results_smote.std():.6f}")
print(f"  Difference - Mean: {cv_differences.mean():.6f}, Std: {cv_differences.std():.6f}")
print(f"  95% CI of difference: [{np.percentile(cv_differences, 2.5):.6f}, {np.percentile(cv_differences, 97.5):.6f}]")

# Check if the difference is stable
pct_folds_exceeding_threshold = (np.abs(cv_differences) > threshold).mean() * 100
print(f"\nFolds where |difference| > {threshold}: {pct_folds_exceeding_threshold:.1f}%")

if pct_folds_exceeding_threshold >= 50:
    stability = "STABLE - finding held up"
else:
    stability = "UNSTABLE - finding did not hold up consistently"

print(f"Stability assessment: {stability}")

# ============================================================================
# 9. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("STEP 9: Save Results")
print("=" * 80)

result = {
    "hypothesis_id": "H5",
    "summary": f"SMOTE oversampling {'DOES' if abs(f1_difference) > threshold else 'DOES NOT'} change the minority-class (>50K) F1 score by more than 0.02 compared to no resampling. The difference in F1 score is {f1_difference:.6f} (SMOTE: {f1_smote:.6f} vs Baseline: {f1_baseline:.6f}), which {'exceeds' if abs(f1_difference) > threshold else 'is within'} the 0.02 threshold.",
    "primary_metric_name": "Minority-class F1 score difference (SMOTE - Baseline)",
    "primary_metric_value": float(f1_difference),
    "direction": f"SMOTE F1 ({f1_smote:.6f}) vs Baseline F1 ({f1_baseline:.6f})",
    "methodological_choices": "Default RandomForestClassifier with no hyperparameter tuning, as specified. Train/test split 80/20 with stratification. Target class '>50K' treated as minority (positive). Categorical features encoded with LabelEncoder. Missing values handled by filling with mode. SMOTE applied only to training data to prevent data leakage.",
    "verification_method": "5 repeated 5-fold stratified cross-validation (25 total folds) on the training data, comparing baseline RF vs RF with SMOTE. Different random seeds used for each repeat to ensure robustness.",
    "verification_result": f"Finding {'HELD UP' if pct_folds_exceeding_threshold >= 50 else 'DID NOT HOLD UP'} - {pct_folds_exceeding_threshold:.1f}% of folds showed |difference| > 0.02. Cross-validation mean difference: {cv_differences.mean():.6f} ± {cv_differences.std():.6f}. 95% CI: [{np.percentile(cv_differences, 2.5):.6f}, {np.percentile(cv_differences, 97.5):.6f}]."
}

with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResults saved to result.json")
print(json.dumps(result, indent=2))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
