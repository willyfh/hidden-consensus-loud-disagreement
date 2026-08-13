#!/usr/bin/env python3
"""
Analysis: Random Forest vs Logistic Regression on Adult Income Dataset
Research Question (H2): Does RF achieve higher stratified 5-fold CV ROC-AUC than LogReg?
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. LOAD AND INSPECT DATA
# ============================================================================
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nColumn names and types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget variable distribution:\n{df['class'].value_counts()}")

# ============================================================================
# 2. DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# Create a copy for preprocessing
df_processed = df.copy()

# Handle missing values (represented as '?' in some datasets)
print("\nChecking for '?' strings that represent missing values...")
for col in df_processed.columns:
    if df_processed[col].dtype == 'object':
        if (df_processed[col] == '?').any():
            print(f"  Column '{col}' has {(df_processed[col] == '?').sum()} missing values")
            df_processed = df_processed[df_processed[col] != '?']

print(f"Dataset shape after removing rows with missing values: {df_processed.shape}")

# Separate features and target
X = df_processed.drop('class', axis=1)
y = df_processed['class'].copy()

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"\nTarget encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
print(f"Target distribution after encoding: {np.bincount(y_encoded)}")

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# Encode categorical variables
X_processed = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

print(f"\nProcessed feature matrix shape: {X_processed.shape}")

# ============================================================================
# 3. STRATIFIED 5-FOLD CROSS-VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STRATIFIED 5-FOLD CROSS-VALIDATION (Main Evaluation)")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# RandomForest with defaults
print("\nRandom Forest (scikit-learn defaults)...")
rf_model = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf_model, X_processed, y_encoded, cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"  ROC-AUC scores (each fold): {rf_scores}")
print(f"  Mean ROC-AUC: {rf_scores.mean():.6f}")
print(f"  Std ROC-AUC: {rf_scores.std():.6f}")

# LogisticRegression with defaults (but scaled features)
print("\nLogistic Regression (scikit-learn defaults)...")
# Scale features for logistic regression
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('logreg', LogisticRegression(random_state=42, max_iter=1000))
])
lr_scores = cross_val_score(pipeline_lr, X_processed, y_encoded, cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"  ROC-AUC scores (each fold): {lr_scores}")
print(f"  Mean ROC-AUC: {lr_scores.mean():.6f}")
print(f"  Std ROC-AUC: {lr_scores.std():.6f}")

# Comparison
roc_auc_difference = rf_scores.mean() - lr_scores.mean()
print(f"\n{'*' * 60}")
print(f"ROC-AUC Difference (RF - LogReg): {roc_auc_difference:.6f}")
print(f"Random Forest: {rf_scores.mean():.6f} ± {rf_scores.std():.6f}")
print(f"Logistic Regression: {lr_scores.mean():.6f} ± {lr_scores.std():.6f}")
print(f"{'*' * 60}")

if roc_auc_difference > 0:
    print(f"✓ Random Forest achieves HIGHER ROC-AUC than Logistic Regression")
else:
    print(f"✗ Logistic Regression achieves HIGHER ROC-AUC than Random Forest")

# ============================================================================
# 4. STABILITY VALIDATION: Repeated Stratified K-Fold
# ============================================================================
print("\n" + "=" * 80)
print("STABILITY VALIDATION: Repeated Stratified 5-Fold CV (10 repetitions)")
print("=" * 80)

rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=None)

print("\nRandom Forest (10 x 5-fold CV with different seeds)...")
rf_scores_repeated = cross_val_score(rf_model, X_processed, y_encoded, cv=rkf, scoring='roc_auc', n_jobs=-1)
print(f"  Number of folds: {len(rf_scores_repeated)}")
print(f"  Mean ROC-AUC: {rf_scores_repeated.mean():.6f}")
print(f"  Std ROC-AUC: {rf_scores_repeated.std():.6f}")
print(f"  Min ROC-AUC: {rf_scores_repeated.min():.6f}")
print(f"  Max ROC-AUC: {rf_scores_repeated.max():.6f}")
print(f"  95% CI: [{np.percentile(rf_scores_repeated, 2.5):.6f}, {np.percentile(rf_scores_repeated, 97.5):.6f}]")

print("\nLogistic Regression (10 x 5-fold CV with different seeds)...")
lr_scores_repeated = cross_val_score(pipeline_lr, X_processed, y_encoded, cv=rkf, scoring='roc_auc', n_jobs=-1)
print(f"  Number of folds: {len(lr_scores_repeated)}")
print(f"  Mean ROC-AUC: {lr_scores_repeated.mean():.6f}")
print(f"  Std ROC-AUC: {lr_scores_repeated.std():.6f}")
print(f"  Min ROC-AUC: {lr_scores_repeated.min():.6f}")
print(f"  Max ROC-AUC: {lr_scores_repeated.max():.6f}")
print(f"  95% CI: [{np.percentile(lr_scores_repeated, 2.5):.6f}, {np.percentile(lr_scores_repeated, 97.5):.6f}]")

# Comparison
roc_auc_difference_repeated = rf_scores_repeated.mean() - lr_scores_repeated.mean()
print(f"\n{'*' * 60}")
print(f"ROC-AUC Difference (RF - LogReg): {roc_auc_difference_repeated:.6f}")
print(f"Random Forest: {rf_scores_repeated.mean():.6f} ± {rf_scores_repeated.std():.6f}")
print(f"Logistic Regression: {lr_scores_repeated.mean():.6f} ± {lr_scores_repeated.std():.6f}")
print(f"{'*' * 60}")

# Check if finding is stable
if (np.percentile(rf_scores_repeated, 2.5) > np.percentile(lr_scores_repeated, 97.5)) or \
   (np.percentile(lr_scores_repeated, 2.5) > np.percentile(rf_scores_repeated, 97.5)):
    print("✓ FINDING IS STABLE: Confidence intervals do not overlap")
else:
    print("⚠ FINDING IS NOT CONCLUSIVE: Confidence intervals overlap")
    if roc_auc_difference_repeated > 0:
        print("  BUT: RF mean is still higher (marginal difference)")
    else:
        print("  AND: LR mean is higher")

# ============================================================================
# 5. SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

print(f"\nDataset size: {df_processed.shape[0]} samples, {X_processed.shape[1]} features")
print(f"Target distribution: {np.bincount(y_encoded)[1]} positive, {np.bincount(y_encoded)[0]} negative")
print(f"\nPrimary finding (5-fold CV):")
print(f"  RF ROC-AUC: {rf_scores.mean():.6f} ± {rf_scores.std():.6f}")
print(f"  LR ROC-AUC: {lr_scores.mean():.6f} ± {lr_scores.std():.6f}")
print(f"  Difference: {roc_auc_difference:.6f}")
print(f"  Direction: {'RF > LogReg' if roc_auc_difference > 0 else 'LogReg > RF'}")

print(f"\nStability check (repeated 5-fold CV, 10 repetitions):")
print(f"  RF ROC-AUC: {rf_scores_repeated.mean():.6f} ± {rf_scores_repeated.std():.6f}")
print(f"  LR ROC-AUC: {lr_scores_repeated.mean():.6f} ± {lr_scores_repeated.std():.6f}")
print(f"  Difference: {roc_auc_difference_repeated:.6f}")
print(f"  Stability: CONFIRMED" if np.sign(roc_auc_difference) == np.sign(roc_auc_difference_repeated) else "  Stability: QUESTIONABLE")

print("\n" + "=" * 80)
print("END OF ANALYSIS")
print("=" * 80)
