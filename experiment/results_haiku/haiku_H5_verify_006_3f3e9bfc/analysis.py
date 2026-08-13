"""
Research Question H5: Does SMOTE oversampling change minority-class F1 score by >0.02?
- Classifier: RandomForestClassifier() with default hyperparameters
- Target: F1 score for minority class (>50K)
- Comparison: SMOTE vs no resampling
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: DATA LOADING AND EXPLORATION")
print("=" * 80)

df = pd.read_csv('adult_income.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nTarget distribution:")
print(df['class'].value_counts())
print(f"Minority class (>50K) proportion: {(df['class'] == '>50K').sum() / len(df):.4f}")

print(f"\nData types and missing values:")
print(df.info())

print(f"\nFirst few rows:")
print(df.head())

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: DATA PREPROCESSING")
print("=" * 80)

# Create a copy for processing
df_processed = df.copy()

# Identify categorical and numerical columns
categorical_cols = df_processed.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove the target column from features
if 'class' in categorical_cols:
    categorical_cols.remove('class')
if 'class' in numerical_cols:
    numerical_cols.remove('class')

print(f"\nNumerical columns: {numerical_cols}")
print(f"Categorical columns: {categorical_cols}")

# Handle missing values in categorical columns
for col in categorical_cols:
    if df_processed[col].isnull().sum() > 0:
        print(f"  - {col}: {df_processed[col].isnull().sum()} missing values -> filling with 'Unknown'")
        df_processed[col].fillna('Unknown', inplace=True)

# Handle missing values in numerical columns
for col in numerical_cols:
    if df_processed[col].isnull().sum() > 0:
        print(f"  - {col}: {df_processed[col].isnull().sum()} missing values -> filling with median")
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

# Encode target
df_processed['class_binary'] = (df_processed['class'] == '>50K').astype(int)
print(f"\nTarget encoded: >50K=1, <=50K=0")

# Encode categorical features using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_processed[col + '_encoded'] = le.fit_transform(df_processed[col].astype(str))
    label_encoders[col] = le

# Prepare feature matrix and target
feature_cols = numerical_cols + [col + '_encoded' for col in categorical_cols]
X = df_processed[feature_cols]
y = df_processed['class_binary']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target distribution:")
print(f"  Class 0 (<=50K): {(y == 0).sum()}")
print(f"  Class 1 (>50K): {(y == 1).sum()}")
print(f"  Imbalance ratio: {(y == 0).sum() / (y == 1).sum():.2f}x")

# ============================================================================
# 3. PRIMARY ANALYSIS: TRAIN/TEST SPLIT COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: PRIMARY ANALYSIS - TRAIN/TEST SPLIT")
print("=" * 80)

# Fixed random state for reproducibility
RANDOM_STATE = 42

# Split data: 70% train, 30% test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train set class distribution: {np.bincount(y_train)}")
print(f"Test set class distribution: {np.bincount(y_test)}")

# ---- Model 1: No resampling ----
print("\n" + "-" * 80)
print("Model 1: Random Forest WITHOUT SMOTE")
print("-" * 80)

rf_no_smote = RandomForestClassifier()  # Default hyperparameters
rf_no_smote.fit(X_train, y_train)
y_pred_no_smote = rf_no_smote.predict(X_test)

f1_no_smote = f1_score(y_test, y_pred_no_smote, pos_label=1)
print(f"F1 Score (minority class >50K): {f1_no_smote:.6f}")

# Get confusion matrix details
from sklearn.metrics import confusion_matrix, classification_report
cm_no_smote = confusion_matrix(y_test, y_pred_no_smote)
print(f"\nConfusion Matrix:")
print(cm_no_smote)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_no_smote, target_names=['<=50K', '>50K']))

# ---- Model 2: With SMOTE ----
print("\n" + "-" * 80)
print("Model 2: Random Forest WITH SMOTE")
print("-" * 80)

# Apply SMOTE to training data only
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training data shape after SMOTE: {X_train_smote.shape}")
print(f"Class distribution after SMOTE: {np.bincount(y_train_smote)}")

rf_smote = RandomForestClassifier()  # Default hyperparameters
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)

f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)
print(f"F1 Score (minority class >50K): {f1_smote:.6f}")

# Get confusion matrix details
cm_smote = confusion_matrix(y_test, y_pred_smote)
print(f"\nConfusion Matrix:")
print(cm_smote)
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_smote, target_names=['<=50K', '>50K']))

# ---- Primary Comparison ----
print("\n" + "=" * 80)
print("PRIMARY RESULT")
print("=" * 80)

f1_difference = abs(f1_smote - f1_no_smote)
print(f"F1 Score WITHOUT SMOTE: {f1_no_smote:.6f}")
print(f"F1 Score WITH SMOTE:    {f1_smote:.6f}")
print(f"Absolute difference:    {f1_difference:.6f}")
print(f"Exceeds 0.02 threshold: {f1_difference > 0.02}")
print(f"SMOTE effect (>50K):    {f1_smote - f1_no_smote:+.6f}")

# ============================================================================
# 4. VALIDATION: REPEATED STRATIFIED K-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: VALIDATION - REPEATED STRATIFIED K-FOLD CV")
print("=" * 80)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

f1_scores_no_smote_cv = []
f1_scores_smote_cv = []
fold_num = 0

print(f"\nRunning 5x5 Repeated Stratified K-Fold CV (25 total folds)...")

for train_idx, val_idx in rskf.split(X, y):
    fold_num += 1

    X_cv_train = X.iloc[train_idx]
    X_cv_val = X.iloc[val_idx]
    y_cv_train = y.iloc[train_idx]
    y_cv_val = y.iloc[val_idx]

    # Model 1: No SMOTE
    rf_no_smote_cv = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_no_smote_cv.fit(X_cv_train, y_cv_train)
    y_pred_no_smote_cv = rf_no_smote_cv.predict(X_cv_val)
    f1_no_smote_cv = f1_score(y_cv_val, y_pred_no_smote_cv, pos_label=1)
    f1_scores_no_smote_cv.append(f1_no_smote_cv)

    # Model 2: With SMOTE
    smote_cv = SMOTE(random_state=RANDOM_STATE)
    X_cv_train_smote, y_cv_train_smote = smote_cv.fit_resample(X_cv_train, y_cv_train)
    rf_smote_cv = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_smote_cv.fit(X_cv_train_smote, y_cv_train_smote)
    y_pred_smote_cv = rf_smote_cv.predict(X_cv_val)
    f1_smote_cv = f1_score(y_cv_val, y_pred_smote_cv, pos_label=1)
    f1_scores_smote_cv.append(f1_smote_cv)

    if fold_num % 5 == 0:
        print(f"  Completed {fold_num} folds...")

f1_scores_no_smote_cv = np.array(f1_scores_no_smote_cv)
f1_scores_smote_cv = np.array(f1_scores_smote_cv)

# Calculate differences for each fold
differences_cv = np.abs(f1_scores_smote_cv - f1_scores_no_smote_cv)
mean_difference_cv = differences_cv.mean()
std_difference_cv = differences_cv.std()

print(f"\nCross-Validation Results (25 folds):")
print(f"  No SMOTE:   Mean F1 = {f1_scores_no_smote_cv.mean():.6f}, Std = {f1_scores_no_smote_cv.std():.6f}")
print(f"  With SMOTE: Mean F1 = {f1_scores_smote_cv.mean():.6f}, Std = {f1_scores_smote_cv.std():.6f}")
print(f"  Mean absolute difference: {mean_difference_cv:.6f} ± {std_difference_cv:.6f}")
print(f"  95% CI for difference: [{mean_difference_cv - 1.96*std_difference_cv:.6f}, {mean_difference_cv + 1.96*std_difference_cv:.6f}]")

# Count folds where difference exceeds 0.02
exceeds_threshold_count = (differences_cv > 0.02).sum()
print(f"  Folds where |difference| > 0.02: {exceeds_threshold_count}/25 ({100*exceeds_threshold_count/25:.1f}%)")

# Direction of effect in CV
smote_better_count = (f1_scores_smote_cv > f1_scores_no_smote_cv).sum()
print(f"  Folds where SMOTE improved F1: {smote_better_count}/25 ({100*smote_better_count/25:.1f}%)")

# ============================================================================
# 5. ADDITIONAL VALIDATION: BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: ADDITIONAL VALIDATION - BOOTSTRAP RESAMPLING")
print("=" * 80)

np.random.seed(RANDOM_STATE)
n_bootstrap = 100
bootstrap_differences = []

print(f"Running {n_bootstrap} bootstrap samples...")

for i in range(n_bootstrap):
    # Bootstrap sample from test set
    indices = np.random.choice(len(X_test), size=len(X_test), replace=True)
    X_boot = X_test.iloc[indices]
    y_boot = y_test.iloc[indices]

    # Predictions from already-trained models
    y_pred_no_smote_boot = rf_no_smote.predict(X_boot)
    y_pred_smote_boot = rf_smote.predict(X_boot)

    # Calculate F1 scores (only if minority class present)
    if (y_boot == 1).sum() > 0:
        f1_no_smote_boot = f1_score(y_boot, y_pred_no_smote_boot, pos_label=1)
        f1_smote_boot = f1_score(y_boot, y_pred_smote_boot, pos_label=1)
        diff_boot = abs(f1_smote_boot - f1_no_smote_boot)
        bootstrap_differences.append(diff_boot)

bootstrap_differences = np.array(bootstrap_differences)

print(f"\nBootstrap Results ({len(bootstrap_differences)} successful samples):")
print(f"  Mean difference: {bootstrap_differences.mean():.6f}")
print(f"  Std dev: {bootstrap_differences.std():.6f}")
print(f"  95% CI: [{np.percentile(bootstrap_differences, 2.5):.6f}, {np.percentile(bootstrap_differences, 97.5):.6f}]")
print(f"  Folds where difference > 0.02: {(bootstrap_differences > 0.02).sum()}/{len(bootstrap_differences)}")

# ============================================================================
# 6. FINAL SUMMARY AND CONCLUSION
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY AND CONCLUSION")
print("=" * 80)

print(f"\nResearch Question:")
print(f"  Does SMOTE oversampling change minority-class (>50K) F1 score by >0.02")
print(f"  compared to no resampling with default RandomForestClassifier?")

print(f"\nPrimary Finding (Train/Test Split):")
print(f"  F1 Score without SMOTE: {f1_no_smote:.6f}")
print(f"  F1 Score with SMOTE:    {f1_smote:.6f}")
print(f"  Absolute difference:    {f1_difference:.6f}")
print(f"  Exceeds 0.02 threshold: {'YES' if f1_difference > 0.02 else 'NO'}")

print(f"\nStability Check (Repeated 5x5-Fold CV):")
print(f"  Mean difference: {mean_difference_cv:.6f} ± {std_difference_cv:.6f}")
print(f"  Folds exceeding 0.02: {exceeds_threshold_count}/25 ({100*exceeds_threshold_count/25:.1f}%)")
print(f"  Stability: {'STABLE' if exceeds_threshold_count > 12 else 'UNSTABLE'}")

print(f"\nBootstrap Validation:")
print(f"  Mean difference: {bootstrap_differences.mean():.6f}")
print(f"  95% CI: [{np.percentile(bootstrap_differences, 2.5):.6f}, {np.percentile(bootstrap_differences, 97.5):.6f}]")

# Overall conclusion
overall_exceeds = (f1_difference > 0.02)
cv_stable = (mean_difference_cv > 0.01)  # Conservative stability check

print(f"\n" + "=" * 80)
if overall_exceeds and cv_stable:
    conclusion = "YES - SMOTE changes F1 score by more than 0.02 (stable across validation)"
elif overall_exceeds:
    conclusion = "YES - SMOTE changes F1 score by more than 0.02 (but unstable)"
elif mean_difference_cv > 0.01:
    conclusion = "BORDERLINE - Initial result <0.02 but CV shows meaningful effect"
else:
    conclusion = "NO - SMOTE does not change F1 score by more than 0.02"

print(f"CONCLUSION: {conclusion}")
print("=" * 80)

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
import json

result = {
    "hypothesis_id": "H5",
    "summary": f"SMOTE oversampling changes the minority-class F1 score by {f1_difference:.6f} ({f1_difference > 0.02 and 'exceeds' or 'does not exceed'} the 0.02 threshold). Repeated cross-validation shows this effect {'holds' if mean_difference_cv > 0.01 else 'does not hold'} consistently.",
    "primary_metric_name": "Absolute F1 score difference (minority class >50K) with vs without SMOTE",
    "primary_metric_value": f1_difference,
    "direction": f"SMOTE {'improves' if f1_smote > f1_no_smote else 'worsens'} F1 by {abs(f1_smote - f1_no_smote):.6f}" + (f" ({'exceeds' if f1_difference > 0.02 else 'below'} 0.02 threshold)" if True else ""),
    "methodological_choices": (
        "1. Train/Test Split: 70/30 stratified split (random_state=42) "
        "2. Preprocessing: LabelEncoder for categorical features, median imputation for numerical features "
        "3. Classifier: RandomForestClassifier() with default hyperparameters (no tuning) "
        "4. SMOTE: Applied only to training data (not test) "
        "5. Metric: F1 score for minority class (>50K, pos_label=1) "
        "6. Imbalance: Original dataset has ~3:1 imbalance (<=50K:>50K) "
        "7. Validation: Repeated Stratified K-Fold (5x5 = 25 folds) plus bootstrap (100 samples)"
    ),
    "verification_method": "Repeated Stratified K-Fold CV (5 splits × 5 repeats = 25 folds, each with independent SMOTE) and bootstrap resampling on held-out test set (100 iterations)",
    "verification_result": (
        f"CV mean difference: {mean_difference_cv:.6f} ± {std_difference_cv:.6f} "
        f"(95% CI: [{mean_difference_cv - 1.96*std_difference_cv:.6f}, {mean_difference_cv + 1.96*std_difference_cv:.6f}]). "
        f"Bootstrap 95% CI: [{np.percentile(bootstrap_differences, 2.5):.6f}, {np.percentile(bootstrap_differences, 97.5):.6f}]. "
        f"Finding {'is stable' if mean_difference_cv > 0.01 else 'is unstable'} - SMOTE effect "
        f"{'persists' if mean_difference_cv > 0.01 else 'does not persist'} in cross-validation."
    )
}

with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResults saved to result.json")
