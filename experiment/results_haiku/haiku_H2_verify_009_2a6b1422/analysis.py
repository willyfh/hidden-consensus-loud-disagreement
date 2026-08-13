"""
Analysis: Random Forest vs Logistic Regression on Adult Income Dataset
Research Question: Does RF achieve higher stratified 5-fold cross-validated ROC-AUC than LogReg?
"""

import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND EXPLORE DATA
# ============================================================================
print("=" * 80)
print("STEP 1: LOADING DATA")
print("=" * 80)

df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['class'].value_counts()}")

# ============================================================================
# 2. PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: PREPROCESSING")
print("=" * 80)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class'].map({'<=50K': 0, '>50K': 1})

print(f"Target distribution: {y.value_counts().to_dict()}")

# Identify column types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")

# Handle missing values
# For numeric: fill with median
# For categorical: fill with mode or 'Unknown'
X_processed = X.copy()

for col in numeric_cols:
    if X_processed[col].isnull().any():
        median_val = X_processed[col].median()
        X_processed[col].fillna(median_val, inplace=True)
        print(f"  Filled {col} missing values with median: {median_val}")

for col in categorical_cols:
    if X_processed[col].isnull().any():
        X_processed[col].fillna('Unknown', inplace=True)
        print(f"  Filled {col} missing values with 'Unknown'")

print(f"\nRemaining missing values: {X_processed.isnull().sum().sum()}")

# Encode categorical variables using LabelEncoder
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col])
    le_dict[col] = le
    print(f"  Encoded {col}: {len(le.classes_)} unique values")

print(f"\nProcessed feature matrix shape: {X_processed.shape}")
print(f"Target shape: {y.shape}")

# ============================================================================
# 3. PRIMARY ANALYSIS: Stratified 5-Fold Cross-Validation
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: PRIMARY ANALYSIS - Stratified 5-Fold CV")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Create models with defaults
rf_model = RandomForestClassifier(random_state=42)
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Scale features for LogReg (good practice for linear models)
# But for fair comparison at this stage, let's use raw features for both
# We'll scale in a pipeline later

# Evaluate RF
print("\nEvaluating Random Forest Classifier...")
rf_scores = cross_val_score(rf_model, X_processed, y, cv=skf, scoring='roc_auc')
rf_mean = rf_scores.mean()
rf_std = rf_scores.std()
print(f"  ROC-AUC scores (5 folds): {rf_scores}")
print(f"  Mean: {rf_mean:.6f}, Std: {rf_std:.6f}")

# Evaluate LogReg
print("\nEvaluating Logistic Regression...")
# Scale features for LogReg
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

lr_scores = cross_val_score(lr_model, X_scaled, y, cv=skf, scoring='roc_auc')
lr_mean = lr_scores.mean()
lr_std = lr_scores.std()
print(f"  ROC-AUC scores (5 folds): {lr_scores}")
print(f"  Mean: {lr_mean:.6f}, Std: {lr_std:.6f}")

# Compare
print("\n" + "-" * 80)
print("PRIMARY COMPARISON:")
print(f"  RF ROC-AUC:  {rf_mean:.6f} ± {rf_std:.6f}")
print(f"  LR ROC-AUC:  {lr_mean:.6f} ± {lr_std:.6f}")
print(f"  Difference:  {rf_mean - lr_mean:+.6f} (RF - LR)")

if rf_mean > lr_mean:
    direction = "RF > LogReg"
    print(f"  => Random Forest OUTPERFORMS Logistic Regression")
else:
    direction = "RF <= LogReg"
    print(f"  => Logistic Regression performs as well or better")

primary_difference = rf_mean - lr_mean

# ============================================================================
# 4. STABILITY VERIFICATION: Repeated Stratified 5-Fold CV
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: STABILITY VERIFICATION - Repeated 5-Fold CV (10 repeats, different seeds)")
print("=" * 80)

n_repeats = 10
all_rf_scores = []
all_lr_scores = []
all_differences = []

for seed in range(n_repeats):
    skf_repeat = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    rf_rep = cross_val_score(rf_model, X_processed, y, cv=skf_repeat, scoring='roc_auc')
    lr_rep = cross_val_score(lr_model, X_scaled, y, cv=skf_repeat, scoring='roc_auc')

    rf_mean_rep = rf_rep.mean()
    lr_mean_rep = lr_rep.mean()
    diff_rep = rf_mean_rep - lr_mean_rep

    all_rf_scores.append(rf_mean_rep)
    all_lr_scores.append(lr_mean_rep)
    all_differences.append(diff_rep)

    print(f"  Seed {seed:2d}: RF={rf_mean_rep:.6f}, LR={lr_mean_rep:.6f}, Diff={diff_rep:+.6f}")

# Summary statistics
all_rf_scores = np.array(all_rf_scores)
all_lr_scores = np.array(all_lr_scores)
all_differences = np.array(all_differences)

print("\n" + "-" * 80)
print("VERIFICATION SUMMARY:")
print(f"  RF mean ROC-AUC:      {all_rf_scores.mean():.6f} ± {all_rf_scores.std():.6f}")
print(f"  LR mean ROC-AUC:      {all_lr_scores.mean():.6f} ± {all_lr_scores.std():.6f}")
print(f"  Mean difference (RF-LR): {all_differences.mean():+.6f}")
print(f"  95% CI of difference: [{np.percentile(all_differences, 2.5):+.6f}, {np.percentile(all_differences, 97.5):+.6f}]")
print(f"  Fraction of repeats where RF > LR: {(all_differences > 0).sum()}/{n_repeats}")

# Determine if finding is stable
if (all_differences > 0).sum() >= 8:  # In at least 8/10 repeats
    verification_result = f"STABLE: RF outperformed LR in {(all_differences > 0).sum()}/10 repeats. Revised estimate: RF ROC-AUC = {all_rf_scores.mean():.6f}, LR ROC-AUC = {all_lr_scores.mean():.6f}, difference = {all_differences.mean():+.6f} (95% CI: [{np.percentile(all_differences, 2.5):+.6f}, {np.percentile(all_differences, 97.5):+.6f}])"
elif (all_differences < 0).sum() >= 8:  # In at least 8/10 repeats
    verification_result = f"STABLE: LR outperformed RF in {(all_differences < 0).sum()}/10 repeats. Revised estimate: RF ROC-AUC = {all_rf_scores.mean():.6f}, LR ROC-AUC = {all_lr_scores.mean():.6f}, difference = {all_differences.mean():+.6f} (95% CI: [{np.percentile(all_differences, 2.5):+.6f}, {np.percentile(all_differences, 97.5):+.6f}])"
else:
    verification_result = f"UNSTABLE: Results vary across repeats. RF outperformed in {(all_differences > 0).sum()}/10 repeats. Mean difference = {all_differences.mean():+.6f} (95% CI: [{np.percentile(all_differences, 2.5):+.6f}, {np.percentile(all_differences, 97.5):+.6f}])"

print(f"\nVerification Result: {verification_result}")

# ============================================================================
# 5. METHODOLOGICAL CHOICES
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: METHODOLOGICAL SUMMARY")
print("=" * 80)

methodological_choices = """
Preprocessing:
- Missing values in numeric columns (fnlwgt) filled with median
- Missing values in categorical columns (workclass, occupation, native-country) filled with 'Unknown'
- Categorical variables encoded using LabelEncoder

Model Configuration:
- Random Forest: scikit-learn defaults (100 estimators, random_state=42)
- Logistic Regression: scikit-learn defaults (max_iter=1000, random_state=42)
- Features scaled for LogReg only (StandardScaler), RF uses raw features

Validation Scheme:
- Primary: Stratified 5-fold cross-validation (stratified by class to maintain ~24%/>50K proportion)
- Verification: 10 repeated stratified 5-fold CV with different random seeds (0-9)
- Shuffle used in verification repeats to ensure robustness

Evaluation Metric:
- ROC-AUC (Receiver Operating Characteristic Area Under Curve)
- Appropriate for imbalanced binary classification (24% vs 76% class split)

Other:
- No class weighting or resampling applied
- No hyperparameter tuning beyond defaults
- fnlwgt (population weight) included as a feature
"""

print(methodological_choices)

# ============================================================================
# 6. PREPARE RESULTS JSON
# ============================================================================
results = {
    "hypothesis_id": "H2",
    "summary": f"Random Forest achieves ROC-AUC of {all_rf_scores.mean():.6f} while Logistic Regression achieves {all_lr_scores.mean():.6f} in stratified 5-fold cross-validation on the Adult income dataset. {'RF outperforms LR' if all_differences.mean() > 0 else 'LR outperforms or matches RF'} with a mean difference of {all_differences.mean():+.6f} (95% CI: [{np.percentile(all_differences, 2.5):+.6f}, {np.percentile(all_differences, 97.5):+.6f}]).",
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(all_differences.mean(), 6),
    "direction": direction,
    "methodological_choices": methodological_choices,
    "verification_method": f"10x repeated stratified 5-fold cross-validation with random seeds 0-9. Each repeat uses shuffled CV folds to ensure stability across different data orderings.",
    "verification_result": verification_result
}

print("\n" + "=" * 80)
print("RESULTS JSON")
print("=" * 80)
print(json.dumps(results, indent=2))

# Save results
with open('result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to result.json")
