"""
Analysis: Does addressing class imbalance improve model quality?
Dataset: Adult Income (UCI/OpenML)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, f1_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')
import json

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("=" * 80)
print("1. LOADING AND EXPLORING DATA")
print("=" * 80)

df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names and types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())

# Check class distribution
print(f"\n" + "=" * 80)
print("CLASS DISTRIBUTION (IMBALANCE CHECK)")
print("=" * 80)
class_dist = df['class'].value_counts()
print(class_dist)
print(f"\nClass proportions:")
print(df['class'].value_counts(normalize=True))
imbalance_ratio = class_dist.max() / class_dist.min()
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print(f"\n" + "=" * 80)
print("2. DATA PREPROCESSING")
print("=" * 80)

# Prepare features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"Categorical columns: {categorical_cols}")

le_dict = {}
X_encoded = X.copy()

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"Target encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

print(f"Encoded features shape: {X_encoded.shape}")
print(f"Features shape after encoding: {X_encoded.shape}")

# ============================================================================
# 3. COMPARE MODELS WITH AND WITHOUT IMBALANCE HANDLING
# ============================================================================
print(f"\n" + "=" * 80)
print("3. MAIN ANALYSIS: MODEL QUALITY WITH VS WITHOUT IMBALANCE HANDLING")
print("=" * 80)

# Define train-test split for initial analysis
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"Train set class distribution:")
print(pd.Series(y_train).value_counts(normalize=True))
print(f"Test set class distribution:")
print(pd.Series(y_test).value_counts(normalize=True))

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# MODEL 1: WITHOUT IMBALANCE HANDLING
# ============================================================================
print(f"\n--- Model 1: WITHOUT Imbalance Handling ---")

# Use Logistic Regression as the base model (interpretable, fast)
model_no_balance = LogisticRegression(max_iter=1000, random_state=42, class_weight=None)
model_no_balance.fit(X_train_scaled, y_train)

y_pred_no_balance = model_no_balance.predict(X_test_scaled)
y_pred_proba_no_balance = model_no_balance.predict_proba(X_test_scaled)[:, 1]

auc_no_balance = roc_auc_score(y_test, y_pred_proba_no_balance)
f1_no_balance = f1_score(y_test, y_pred_no_balance)
balanced_acc_no_balance = balanced_accuracy_score(y_test, y_pred_no_balance)

print(f"ROC-AUC: {auc_no_balance:.4f}")
print(f"F1-Score: {f1_no_balance:.4f}")
print(f"Balanced Accuracy: {balanced_acc_no_balance:.4f}")

# ============================================================================
# MODEL 2: WITH IMBALANCE HANDLING (SMOTE)
# ============================================================================
print(f"\n--- Model 2: WITH Imbalance Handling (SMOTE) ---")

# Apply SMOTE on training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"Training set after SMOTE:")
print(f"Class distribution: {np.bincount(y_train_smote)}")
print(f"Proportions: {np.bincount(y_train_smote) / len(y_train_smote)}")

# Train model with SMOTE-balanced data
model_with_smote = LogisticRegression(max_iter=1000, random_state=42, class_weight=None)
model_with_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = model_with_smote.predict(X_test_scaled)
y_pred_proba_smote = model_with_smote.predict_proba(X_test_scaled)[:, 1]

auc_smote = roc_auc_score(y_test, y_pred_proba_smote)
f1_smote = f1_score(y_test, y_pred_smote)
balanced_acc_smote = balanced_accuracy_score(y_test, y_pred_smote)

print(f"ROC-AUC: {auc_smote:.4f}")
print(f"F1-Score: {f1_smote:.4f}")
print(f"Balanced Accuracy: {balanced_acc_smote:.4f}")

# ============================================================================
# COMPARISON
# ============================================================================
print(f"\n" + "=" * 80)
print("COMPARISON: WITH vs WITHOUT IMBALANCE HANDLING")
print("=" * 80)

comparison_data = {
    'Metric': ['ROC-AUC', 'F1-Score', 'Balanced Accuracy'],
    'Without Balance': [auc_no_balance, f1_no_balance, balanced_acc_no_balance],
    'With SMOTE': [auc_smote, f1_smote, balanced_acc_smote]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Difference (SMOTE - No Balance)'] = comparison_df['With SMOTE'] - comparison_df['Without Balance']
comparison_df['% Improvement'] = (comparison_df['Difference (SMOTE - No Balance)'] / comparison_df['Without Balance'] * 100).round(2)

print(comparison_df.to_string(index=False))

# Primary metric: ROC-AUC (most reliable for imbalanced datasets)
primary_metric_diff = auc_smote - auc_no_balance
print(f"\nPrimary Metric (ROC-AUC) Improvement: {primary_metric_diff:.4f}")

# ============================================================================
# 4. VALIDATION: REPEATED CROSS-VALIDATION
# ============================================================================
print(f"\n" + "=" * 80)
print("4. VALIDATION: REPEATED 5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 80)
print("Testing stability across different data splits and random seeds...")

def evaluate_with_cv(X_data, y_data, use_smote=False, cv_folds=5, n_repeats=5):
    """
    Perform repeated stratified cross-validation
    """
    all_auc_scores = []
    all_f1_scores = []
    all_balanced_acc_scores = []

    for repeat_idx in range(n_repeats):
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=repeat_idx)

        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_data, y_data)):
            X_cv_train = X_data.iloc[train_idx]
            X_cv_test = X_data.iloc[test_idx]
            y_cv_train = y_data.iloc[train_idx]
            y_cv_test = y_data.iloc[test_idx]

            # Scale
            scaler_cv = StandardScaler()
            X_cv_train_scaled = scaler_cv.fit_transform(X_cv_train)
            X_cv_test_scaled = scaler_cv.transform(X_cv_test)

            # Apply SMOTE if requested
            if use_smote:
                smote_cv = SMOTE(random_state=repeat_idx)
                X_cv_train_scaled, y_cv_train = smote_cv.fit_resample(X_cv_train_scaled, y_cv_train.values)

            # Train and evaluate
            model = LogisticRegression(max_iter=1000, random_state=repeat_idx, class_weight=None)
            model.fit(X_cv_train_scaled, y_cv_train)

            y_pred = model.predict(X_cv_test_scaled)
            y_pred_proba = model.predict_proba(X_cv_test_scaled)[:, 1]

            auc_cv = roc_auc_score(y_cv_test, y_pred_proba)
            f1_cv = f1_score(y_cv_test, y_pred)
            balanced_acc_cv = balanced_accuracy_score(y_cv_test, y_pred)

            all_auc_scores.append(auc_cv)
            all_f1_scores.append(f1_cv)
            all_balanced_acc_scores.append(balanced_acc_cv)

    return {
        'auc': np.array(all_auc_scores),
        'f1': np.array(all_f1_scores),
        'balanced_acc': np.array(all_balanced_acc_scores)
    }

# Run CV for both approaches
cv_results_no_balance = evaluate_with_cv(X_encoded, pd.Series(y_encoded), use_smote=False, cv_folds=5, n_repeats=5)
cv_results_with_balance = evaluate_with_cv(X_encoded, pd.Series(y_encoded), use_smote=True, cv_folds=5, n_repeats=5)

print(f"\nCross-Validation Results (25 folds total: 5 repeats × 5 folds):")
print(f"\n--- WITHOUT Imbalance Handling ---")
print(f"ROC-AUC:          Mean={cv_results_no_balance['auc'].mean():.4f}, Std={cv_results_no_balance['auc'].std():.4f}")
print(f"                  95% CI: [{np.percentile(cv_results_no_balance['auc'], 2.5):.4f}, {np.percentile(cv_results_no_balance['auc'], 97.5):.4f}]")
print(f"F1-Score:         Mean={cv_results_no_balance['f1'].mean():.4f}, Std={cv_results_no_balance['f1'].std():.4f}")
print(f"Balanced Accuracy: Mean={cv_results_no_balance['balanced_acc'].mean():.4f}, Std={cv_results_no_balance['balanced_acc'].std():.4f}")

print(f"\n--- WITH Imbalance Handling (SMOTE) ---")
print(f"ROC-AUC:          Mean={cv_results_with_balance['auc'].mean():.4f}, Std={cv_results_with_balance['auc'].std():.4f}")
print(f"                  95% CI: [{np.percentile(cv_results_with_balance['auc'], 2.5):.4f}, {np.percentile(cv_results_with_balance['auc'], 97.5):.4f}]")
print(f"F1-Score:         Mean={cv_results_with_balance['f1'].mean():.4f}, Std={cv_results_with_balance['f1'].std():.4f}")
print(f"Balanced Accuracy: Mean={cv_results_with_balance['balanced_acc'].mean():.4f}, Std={cv_results_with_balance['balanced_acc'].std():.4f}")

print(f"\n--- DIFFERENCES (With - Without) ---")
auc_cv_diff = cv_results_with_balance['auc'].mean() - cv_results_no_balance['auc'].mean()
f1_cv_diff = cv_results_with_balance['f1'].mean() - cv_results_no_balance['f1'].mean()
balanced_acc_cv_diff = cv_results_with_balance['balanced_acc'].mean() - cv_results_no_balance['balanced_acc'].mean()

print(f"ROC-AUC Difference:          {auc_cv_diff:.4f}")
print(f"F1-Score Difference:         {f1_cv_diff:.4f}")
print(f"Balanced Accuracy Difference: {balanced_acc_cv_diff:.4f}")

# Statistical significance check (do confidence intervals overlap?)
ci_no_balance = [np.percentile(cv_results_no_balance['auc'], 2.5),
                  np.percentile(cv_results_no_balance['auc'], 97.5)]
ci_with_balance = [np.percentile(cv_results_with_balance['auc'], 2.5),
                   np.percentile(cv_results_with_balance['auc'], 97.5)]

print(f"\nConfidence Interval Overlap (ROC-AUC):")
print(f"Without Balance CI: {ci_no_balance}")
print(f"With Balance CI:    {ci_with_balance}")
ci_overlap = not (ci_with_balance[0] > ci_no_balance[1] or ci_no_balance[0] > ci_with_balance[1])
print(f"CIs overlap: {ci_overlap}")

# ============================================================================
# 5. FINAL SUMMARY
# ============================================================================
print(f"\n" + "=" * 80)
print("5. FINAL SUMMARY")
print("=" * 80)

# Determine the finding
auc_improvement_held = auc_cv_diff > 0
magnitude = abs(auc_cv_diff)

if auc_improvement_held and magnitude > 0.001:  # meaningful improvement
    finding = "YES: Addressing class imbalance (via SMOTE) improves model quality"
    direction = "SMOTE improves ROC-AUC"
elif auc_improvement_held and magnitude <= 0.001:
    finding = "MINIMAL: Marginal improvement when addressing class imbalance"
    direction = "Slight improvement with SMOTE, but negligible"
else:
    finding = "NO: Addressing class imbalance does not improve model quality"
    direction = "No SMOTE improves or maintains performance"

print(f"\nFinding: {finding}")
print(f"Direction: {direction}")
print(f"\nCV AUC Improvement (Primary Metric): {auc_cv_diff:.4f}")
print(f"Initial test set improvement: {primary_metric_diff:.4f}")
print(f"Stability check: Finding {'HOLDS' if auc_improvement_held else 'REVERSED'} under 5×5 CV")

# ============================================================================
# 6. SAVE RESULTS
# ============================================================================
result = {
    "hypothesis_id": "H4",
    "summary": finding,
    "primary_metric_name": "ROC-AUC difference (SMOTE - No Balance)",
    "primary_metric_value": float(auc_cv_diff),
    "direction": direction,
    "methodological_choices": (
        "Logistic Regression model with standard scaling. Class imbalance handled via SMOTE (Synthetic Minority Over-sampling). "
        "Primary metric: ROC-AUC (robust to class imbalance). Secondary metrics: F1-Score, Balanced Accuracy. "
        "Validation: 5 repeats of 5-fold stratified cross-validation with different random seeds (25 total folds). "
        "Train-test split: 70-30 with stratification."
    ),
    "verification_method": "5 repeats × 5-fold stratified cross-validation with different random seeds (25 total folds)",
    "verification_result": (
        f"Finding {'CONFIRMED' if auc_improvement_held else 'NOT CONFIRMED'}. "
        f"Cross-validation ROC-AUC without SMOTE: {cv_results_no_balance['auc'].mean():.4f} ± {cv_results_no_balance['auc'].std():.4f}. "
        f"Cross-validation ROC-AUC with SMOTE: {cv_results_with_balance['auc'].mean():.4f} ± {cv_results_with_balance['auc'].std():.4f}. "
        f"Mean difference: {auc_cv_diff:.4f}. "
        f"95% CIs {'overlap' if ci_overlap else 'do not overlap'}, suggesting "
        f"{'no statistically significant difference' if ci_overlap else 'a significant improvement'}."
    )
}

with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n✓ Results saved to result.json")
print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
