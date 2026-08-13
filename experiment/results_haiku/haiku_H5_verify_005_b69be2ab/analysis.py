"""
Analysis: Effect of SMOTE oversampling on minority-class F1 score
Research Question H5: Does SMOTE change minority-class (>50K) F1 by > 0.02?
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("="*70)
print("LOADING DATA")
print("="*70)
df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nTarget class distribution:")
print(df['class'].value_counts())
print(f"Proportion of minority class (>50K): {(df['class'] == '>50K').sum() / len(df):.4f}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Make a copy for processing
data = df.copy()

# Handle missing values: drop rows with any missing values in features
# (we'll keep target as is and drop if missing)
data = data.dropna(subset=['class'])
initial_rows = len(data)
data = data.dropna()
print(f"Rows after dropping missing values: {len(data)} (dropped {initial_rows - len(data)})")

# Separate features and target
X = data.drop('class', axis=1)
y = data['class'].apply(lambda x: 1 if x == '>50K' else 0)  # 1 = >50K, 0 = <=50K

print(f"\nTarget distribution after cleaning:")
print(f"  Class 0 (<=50K): {(y == 0).sum()}")
print(f"  Class 1 (>50K): {(y == 1).sum()}")
print(f"  Imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}:1")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Encode categorical variables using LabelEncoder
X_encoded = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

print(f"\nEncoded {len(categorical_cols)} categorical columns")

# ============================================================================
# 3. TRAIN-TEST SPLIT
# ============================================================================
print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Training set class distribution:")
print(f"  Class 0: {(y_train == 0).sum()}")
print(f"  Class 1: {(y_train == 1).sum()}")

# ============================================================================
# 4. BASELINE: NO RESAMPLING
# ============================================================================
print("\n" + "="*70)
print("BASELINE: NO RESAMPLING")
print("="*70)

rf_baseline = RandomForestClassifier(random_state=RANDOM_STATE)
rf_baseline.fit(X_train, y_train)

y_pred_baseline = rf_baseline.predict(X_test)
f1_baseline = f1_score(y_test, y_pred_baseline, pos_label=1)

print(f"F1 Score (>50K / minority class): {f1_baseline:.6f}")
print(f"\nClassification report (baseline):")
print(classification_report(y_test, y_pred_baseline, target_names=['<=50K', '>50K']))

# ============================================================================
# 5. WITH SMOTE OVERSAMPLING
# ============================================================================
print("\n" + "="*70)
print("WITH SMOTE OVERSAMPLING")
print("="*70)

# Apply SMOTE only to training data
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set after SMOTE:")
print(f"  Class 0: {(y_train_smote == 0).sum()}")
print(f"  Class 1: {(y_train_smote == 1).sum()}")

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = rf_smote.predict(X_test)
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)

print(f"F1 Score (>50K / minority class): {f1_smote:.6f}")
print(f"\nClassification report (SMOTE):")
print(classification_report(y_test, y_pred_smote, target_names=['<=50K', '>50K']))

# ============================================================================
# 6. COMPARE RESULTS
# ============================================================================
print("\n" + "="*70)
print("PRIMARY FINDING: SMOTE EFFECT ON MINORITY-CLASS F1")
print("="*70)

f1_difference = f1_smote - f1_baseline
exceeds_threshold = abs(f1_difference) > 0.02

print(f"F1 Score (No Resampling):  {f1_baseline:.6f}")
print(f"F1 Score (With SMOTE):     {f1_smote:.6f}")
print(f"Absolute Difference:       {f1_difference:.6f}")
print(f"Exceeds 0.02 threshold?    {exceeds_threshold}")

if f1_difference > 0:
    direction = f"SMOTE IMPROVES F1 by {f1_difference:.6f}"
elif f1_difference < 0:
    direction = f"SMOTE DECREASES F1 by {abs(f1_difference):.6f}"
else:
    direction = "NO CHANGE"

print(f"Direction: {direction}")

# ============================================================================
# 7. STABILITY VALIDATION: REPEATED STRATIFIED K-FOLD CV
# ============================================================================
print("\n" + "="*70)
print("STABILITY VALIDATION: REPEATED STRATIFIED K-FOLD CV")
print("="*70)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

f1_scores_baseline = []
f1_scores_smote = []
f1_diffs = []

fold_num = 0
for train_idx, val_idx in rskf.split(X_encoded, y):
    fold_num += 1

    X_fold_train = X_encoded.iloc[train_idx]
    X_fold_val = X_encoded.iloc[val_idx]
    y_fold_train = y.iloc[train_idx]
    y_fold_val = y.iloc[val_idx]

    # Baseline (no resampling)
    rf_fold_base = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_fold_base.fit(X_fold_train, y_fold_train)
    y_pred_fold_base = rf_fold_base.predict(X_fold_val)
    f1_fold_base = f1_score(y_fold_val, y_pred_fold_base, pos_label=1)
    f1_scores_baseline.append(f1_fold_base)

    # With SMOTE
    smote_fold = SMOTE(random_state=RANDOM_STATE)
    X_fold_train_smote, y_fold_train_smote = smote_fold.fit_resample(X_fold_train, y_fold_train)
    rf_fold_smote = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_fold_smote.fit(X_fold_train_smote, y_fold_train_smote)
    y_pred_fold_smote = rf_fold_smote.predict(X_fold_val)
    f1_fold_smote = f1_score(y_fold_val, y_pred_fold_smote, pos_label=1)
    f1_scores_smote.append(f1_fold_smote)

    # Difference
    diff = f1_fold_smote - f1_fold_base
    f1_diffs.append(diff)

print(f"Completed {fold_num} folds (5-fold x 5 repeats)")

f1_scores_baseline = np.array(f1_scores_baseline)
f1_scores_smote = np.array(f1_scores_smote)
f1_diffs = np.array(f1_diffs)

print(f"\nBaseline F1 Scores (No Resampling):")
print(f"  Mean:   {f1_scores_baseline.mean():.6f}")
print(f"  Std:    {f1_scores_baseline.std():.6f}")
print(f"  Min:    {f1_scores_baseline.min():.6f}")
print(f"  Max:    {f1_scores_baseline.max():.6f}")

print(f"\nSMOTE F1 Scores:")
print(f"  Mean:   {f1_scores_smote.mean():.6f}")
print(f"  Std:    {f1_scores_smote.std():.6f}")
print(f"  Min:    {f1_scores_smote.min():.6f}")
print(f"  Max:    {f1_scores_smote.max():.6f}")

print(f"\nF1 Score Differences (SMOTE - Baseline):")
print(f"  Mean Difference:        {f1_diffs.mean():.6f}")
print(f"  Std of Differences:     {f1_diffs.std():.6f}")
print(f"  Min Difference:         {f1_diffs.min():.6f}")
print(f"  Max Difference:         {f1_diffs.max():.6f}")
print(f"  95% CI:                 [{np.percentile(f1_diffs, 2.5):.6f}, {np.percentile(f1_diffs, 97.5):.6f}]")

# Stability check
mean_diff_cv = f1_diffs.mean()
std_diff_cv = f1_diffs.std()
exceeds_threshold_cv = abs(mean_diff_cv) > 0.02

print(f"\n  Mean difference exceeds 0.02 threshold? {exceeds_threshold_cv}")
print(f"  Direction consistent across folds? {(f1_diffs > 0).sum()}/{len(f1_diffs)} folds show SMOTE improvement")

# ============================================================================
# 8. FINAL CONCLUSION
# ============================================================================
print("\n" + "="*70)
print("FINAL CONCLUSION")
print("="*70)

print(f"\nPrimary Test-Set Finding:")
print(f"  F1 Difference: {f1_difference:.6f}")
print(f"  Exceeds 0.02?  {exceeds_threshold}")

print(f"\nValidation (CV) Finding:")
print(f"  Mean CV F1 Difference: {mean_diff_cv:.6f}")
print(f"  Exceeds 0.02?          {exceeds_threshold_cv}")

if abs(f1_difference) > 0.02 and abs(mean_diff_cv) > 0.02:
    conclusion = f"YES - SMOTE {'improves' if f1_difference > 0 else 'decreases'} minority-class F1 by more than 0.02 (Test: {f1_difference:.6f}, CV: {mean_diff_cv:.6f})"
elif abs(f1_difference) > 0.02 and abs(mean_diff_cv) <= 0.02:
    conclusion = f"UNSTABLE - Test set shows change > 0.02, but CV validation does not (Test: {f1_difference:.6f}, CV: {mean_diff_cv:.6f})"
elif abs(f1_difference) <= 0.02 and abs(mean_diff_cv) > 0.02:
    conclusion = f"UNSTABLE - CV shows change > 0.02, but test set does not (Test: {f1_difference:.6f}, CV: {mean_diff_cv:.6f})"
else:
    conclusion = f"NO - Neither test ({f1_difference:.6f}) nor CV ({mean_diff_cv:.6f}) exceed 0.02 threshold"

print(f"\nConclusion: {conclusion}")

# ============================================================================
# 9. PREPARE RESULT JSON
# ============================================================================
result = {
    "hypothesis_id": "H5",
    "summary": f"SMOTE oversampling changes the minority-class (>50K) F1 score by {f1_difference:.6f} on the test set, which {'exceeds' if exceeds_threshold else 'does not exceed'} the 0.02 threshold. Cross-validation validation (mean difference: {mean_diff_cv:.6f}) {'confirms' if abs(mean_diff_cv) > 0.02 else 'does not confirm'} this finding.",
    "primary_metric_name": "F1 Score difference (SMOTE - No Resampling) for minority class (>50K)",
    "primary_metric_value": round(f1_difference, 6),
    "direction": f"{'SMOTE improves' if f1_difference > 0 else 'SMOTE decreases'} F1 by {abs(f1_difference):.6f} ({'exceeds' if exceeds_threshold else 'does not exceed'} 0.02 threshold)",
    "methodological_choices": (
        "Preprocessing: Dropped rows with missing values. Encoded all categorical features using LabelEncoder. "
        "Train-test split: 80-20 stratified split (random_state=42). "
        "Model: Default RandomForestClassifier(random_state=42) for both conditions. "
        "SMOTE: Applied only to training data using imblearn.over_sampling.SMOTE(random_state=42), using default k_neighbors=5. "
        "Evaluation: F1 score on test set, focusing on minority class (>50K) as pos_label=1. "
        "Target encoding: 1 = >50K (minority), 0 = <=50K (majority)."
    ),
    "verification_method": "Repeated Stratified K-Fold Cross-Validation: 5-fold split repeated 5 times (25 total folds) with stratification to maintain class balance. Each fold evaluated for both baseline and SMOTE conditions.",
    "verification_result": f"Finding {'STABLE' if abs(f1_difference) > 0.02 and abs(mean_diff_cv) > 0.02 else 'UNSTABLE' if (abs(f1_difference) > 0.02) != (abs(mean_diff_cv) > 0.02) else 'NEGATIVE'}. Test-set F1 difference: {f1_difference:.6f}. CV mean difference: {mean_diff_cv:.6f} (std={std_diff_cv:.6f}, 95% CI=[{np.percentile(f1_diffs, 2.5):.6f}, {np.percentile(f1_diffs, 97.5):.6f}]). {(f1_diffs > 0).sum()}/{len(f1_diffs)} CV folds show improvement with SMOTE."
}

# Save result to JSON
import json
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\n✓ Result saved to result.json")
print(f"\nResult summary:")
print(json.dumps(result, indent=2))
