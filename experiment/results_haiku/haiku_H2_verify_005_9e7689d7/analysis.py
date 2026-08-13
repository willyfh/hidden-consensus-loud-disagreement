"""
Analysis: RF vs LogReg on Adult Income Classification
Research Question: Does RF achieve higher stratified 5-fold CV ROC-AUC than LogReg?
"""

import pandas as pd
import numpy as np
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, RepeatedStratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load data
print("Loading data...")
df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['class'].value_counts()}")
print()

# Preprocessing
print("Preprocessing data...")

# Handle target variable
df['class'] = (df['class'] == '>50K').astype(int)
y = df['class'].values

# Select features (all except class)
X_raw = df.drop('class', axis=1)

# Identify numeric and categorical columns
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_raw.select_dtypes(exclude=[np.number]).columns.tolist()

print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")
print(f"Categorical columns ({len(categorical_cols)}): {categorical_cols}")
print()

# Handle missing values
# For categorical: replace with 'missing'
for col in categorical_cols:
    X_raw[col] = X_raw[col].fillna('missing')

# For numeric: replace with median
for col in numeric_cols:
    X_raw[col] = X_raw[col].fillna(X_raw[col].median())

print("Missing values after imputation:")
print(X_raw.isnull().sum().sum())
print()

# Encode categorical features using LabelEncoder
X_encoded = X_raw.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le

X = X_encoded.values
print(f"Feature matrix shape: {X.shape}")
print()

# ============================================================================
# PRIMARY ANALYSIS: Stratified 5-Fold Cross-Validation
# ============================================================================
print("="*70)
print("PRIMARY ANALYSIS: Stratified 5-Fold Cross-Validation")
print("="*70)

# Initialize stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest (scikit-learn defaults)
print("\nTraining Random Forest with stratified 5-fold CV...")
rf = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
rf_mean = rf_scores.mean()
rf_std = rf_scores.std()
print(f"RF ROC-AUC scores: {rf_scores}")
print(f"RF ROC-AUC mean: {rf_mean:.6f} (+/- {rf_std:.6f})")

# Logistic Regression (scikit-learn defaults)
print("\nTraining Logistic Regression with stratified 5-fold CV...")
# Note: LogReg needs scaled features for best performance, but we'll use raw encoding
# to avoid introducing extra preprocessing that could bias the comparison
lr = LogisticRegression(max_iter=1000, random_state=42)
lr_scores = cross_val_score(lr, X, y, cv=skf, scoring='roc_auc', n_jobs=-1)
lr_mean = lr_scores.mean()
lr_std = lr_scores.std()
print(f"LR ROC-AUC scores: {lr_scores}")
print(f"LR ROC-AUC mean: {lr_mean:.6f} (+/- {lr_std:.6f})")

# Calculate difference
auc_diff = rf_mean - lr_mean
print(f"\nROC-AUC Difference (RF - LR): {auc_diff:.6f}")
print(f"RF > LR: {rf_mean > lr_mean}")
print()

# ============================================================================
# VALIDATION: Repeated Stratified 5-Fold Cross-Validation
# ============================================================================
print("="*70)
print("VALIDATION: Repeated Stratified 5-Fold Cross-Validation (5 repeats)")
print("="*70)

# Repeated stratified k-fold with different random states
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

print("\nTraining Random Forest with repeated stratified 5-fold CV...")
rf_repeated_scores = cross_val_score(rf, X, y, cv=rskf, scoring='roc_auc', n_jobs=-1)
rf_repeated_mean = rf_repeated_scores.mean()
rf_repeated_std = rf_repeated_scores.std()
print(f"RF ROC-AUC mean (5x5 repeats): {rf_repeated_mean:.6f} (+/- {rf_repeated_std:.6f})")
print(f"RF ROC-AUC range: [{rf_repeated_scores.min():.6f}, {rf_repeated_scores.max():.6f}]")

print("\nTraining Logistic Regression with repeated stratified 5-fold CV...")
lr_repeated_scores = cross_val_score(lr, X, y, cv=rskf, scoring='roc_auc', n_jobs=-1)
lr_repeated_mean = lr_repeated_scores.mean()
lr_repeated_std = lr_repeated_scores.std()
print(f"LR ROC-AUC mean (5x5 repeats): {lr_repeated_mean:.6f} (+/- {lr_repeated_std:.6f})")
print(f"LR ROC-AUC range: [{lr_repeated_scores.min():.6f}, {lr_repeated_scores.max():.6f}]")

# Repeated difference
auc_diff_repeated = rf_repeated_mean - lr_repeated_mean
print(f"\nROC-AUC Difference (RF - LR, repeated): {auc_diff_repeated:.6f}")
print(f"RF > LR (repeated): {rf_repeated_mean > lr_repeated_mean}")
print()

# ============================================================================
# STABILITY CHECK: Fold-wise comparison
# ============================================================================
print("="*70)
print("STABILITY CHECK: Fold-wise Comparison")
print("="*70)

fold_wins = (rf_scores > lr_scores).sum()
print(f"\nNumber of folds where RF > LR: {fold_wins} / 5")
print(f"Folds where RF wins: {np.where(rf_scores > lr_scores)[0] + 1}")
print()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("="*70)
print("SUMMARY STATISTICS")
print("="*70)
print(f"\nPrimary finding (5-fold CV):")
print(f"  RF ROC-AUC:  {rf_mean:.6f}")
print(f"  LR ROC-AUC:  {lr_mean:.6f}")
print(f"  Difference:  {auc_diff:.6f}")
print(f"  RF > LR:     {rf_mean > lr_mean}")

print(f"\nValidation finding (5x5 repeated CV):")
print(f"  RF ROC-AUC:  {rf_repeated_mean:.6f}")
print(f"  LR ROC-AUC:  {lr_repeated_mean:.6f}")
print(f"  Difference:  {auc_diff_repeated:.6f}")
print(f"  RF > LR:     {rf_repeated_mean > lr_repeated_mean}")

print(f"\nStability:")
print(f"  Primary diff matches validation direction: {(auc_diff > 0) == (auc_diff_repeated > 0)}")
print(f"  Primary diff magnitude: {auc_diff:.6f}")
print(f"  Validation diff magnitude: {auc_diff_repeated:.6f}")

# Prepare output
output = {
    "primary_rf_auc": float(rf_mean),
    "primary_lr_auc": float(lr_mean),
    "primary_difference": float(auc_diff),
    "rf_wins": bool(rf_mean > lr_mean),
    "validation_rf_auc": float(rf_repeated_mean),
    "validation_lr_auc": float(lr_repeated_mean),
    "validation_difference": float(auc_diff_repeated),
    "validation_rf_wins": bool(rf_repeated_mean > lr_repeated_mean),
    "fold_wins_count": int(fold_wins),
    "total_folds": 5,
}

print("\nAnalysis complete.")
