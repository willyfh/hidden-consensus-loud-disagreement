"""
Analysis: Does addressing class imbalance improve model quality?
Dataset: Adult income classification
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_curve, auc
)
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("CLASS IMBALANCE ANALYSIS ON ADULT INCOME DATASET")
print("=" * 70)

# Load data
df = pd.read_csv('adult_income.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Class distribution:\n{df['class'].value_counts(normalize=True)}")

# Preprocessing
# Handle missing values
df['workclass'].fillna(df['workclass'].mode()[0], inplace=True)
df['occupation'].fillna(df['occupation'].mode()[0], inplace=True)
df['native-country'].fillna(df['native-country'].mode()[0], inplace=True)

# Drop fnlwgt (survey weight, not a feature)
X = df.drop(['class', 'fnlwgt'], axis=1)
y = (df['class'] == '>50K').astype(int)

# Encode categorical features
le_dict = {}
for col in X.select_dtypes(include='object').columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

print(f"\nFeatures after preprocessing: {X.shape[1]}")
print(f"Target distribution: {y.value_counts().to_dict()}")
print(f"Class imbalance ratio: {(1 - y.mean()) / y.mean():.2f}:1 (negative:positive)")

# Train-test split (stratified to maintain class distribution in test set)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples, {y_train.mean():.2%} positive class")
print(f"Test set: {X_test.shape[0]} samples, {y_test.mean():.2%} positive class")

# ============================================================================
# EXPERIMENT 1: BASELINE MODEL (NO IMBALANCE HANDLING)
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 1: BASELINE (NO IMBALANCE HANDLING)")
print("=" * 70)

models_baseline = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

baseline_results = {}
for name, model in models_baseline.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    baseline_results[name] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
    }

    print(f"\n{name}:")
    print(f"  ROC-AUC: {baseline_results[name]['roc_auc']:.4f}")
    print(f"  F1-Score: {baseline_results[name]['f1']:.4f}")
    print(f"  Precision: {baseline_results[name]['precision']:.4f}")
    print(f"  Recall: {baseline_results[name]['recall']:.4f}")
    print(f"  Balanced Accuracy: {baseline_results[name]['balanced_accuracy']:.4f}")

# ============================================================================
# EXPERIMENT 2: CLASS WEIGHTS HANDLING
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 2: CLASS WEIGHTS")
print("=" * 70)

models_weighted = {
    'LogisticRegression_weighted': LogisticRegression(
        max_iter=1000, random_state=42, class_weight='balanced'
    ),
    'RandomForest_weighted': RandomForestClassifier(
        n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1
    )
}

weighted_results = {}
for name, model in models_weighted.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    weighted_results[name] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
    }

    print(f"\n{name}:")
    print(f"  ROC-AUC: {weighted_results[name]['roc_auc']:.4f}")
    print(f"  F1-Score: {weighted_results[name]['f1']:.4f}")
    print(f"  Precision: {weighted_results[name]['precision']:.4f}")
    print(f"  Recall: {weighted_results[name]['recall']:.4f}")
    print(f"  Balanced Accuracy: {weighted_results[name]['balanced_accuracy']:.4f}")

# ============================================================================
# EXPERIMENT 3: OVERSAMPLING (SMOTE)
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 3: SMOTE OVERSAMPLING")
print("=" * 70)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
print(f"\nAfter SMOTE: {X_train_smote.shape[0]} samples, {y_train_smote.mean():.2%} positive class")

models_smote = {
    'LogisticRegression_SMOTE': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest_SMOTE': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

smote_results = {}
for name, model in models_smote.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    smote_results[name] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
    }

    print(f"\n{name}:")
    print(f"  ROC-AUC: {smote_results[name]['roc_auc']:.4f}")
    print(f"  F1-Score: {smote_results[name]['f1']:.4f}")
    print(f"  Precision: {smote_results[name]['precision']:.4f}")
    print(f"  Recall: {smote_results[name]['recall']:.4f}")
    print(f"  Balanced Accuracy: {smote_results[name]['balanced_accuracy']:.4f}")

# ============================================================================
# EXPERIMENT 4: STRATIFIED RANDOM UNDERSAMPLING
# ============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT 4: RANDOM UNDERSAMPLING")
print("=" * 70)

rus = RandomUnderSampler(random_state=42)
X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)
print(f"\nAfter undersampling: {X_train_rus.shape[0]} samples, {y_train_rus.mean():.2%} positive class")

models_rus = {
    'LogisticRegression_RUS': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest_RUS': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

rus_results = {}
for name, model in models_rus.items():
    model.fit(X_train_rus, y_train_rus)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    rus_results[name] = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_test, y_pred)
    }

    print(f"\n{name}:")
    print(f"  ROC-AUC: {rus_results[name]['roc_auc']:.4f}")
    print(f"  F1-Score: {rus_results[name]['f1']:.4f}")
    print(f"  Precision: {rus_results[name]['precision']:.4f}")
    print(f"  Recall: {rus_results[name]['recall']:.4f}")
    print(f"  Balanced Accuracy: {rus_results[name]['balanced_accuracy']:.4f}")

# ============================================================================
# COMPARATIVE ANALYSIS
# ============================================================================
print("\n" + "=" * 70)
print("COMPARATIVE SUMMARY")
print("=" * 70)

# Aggregate all results
all_results = {
    **baseline_results,
    **weighted_results,
    **smote_results,
    **rus_results
}

# Compute improvements
print("\n--- ROC-AUC Metric (primary) ---")
baseline_auc_lr = baseline_results['LogisticRegression']['roc_auc']
baseline_auc_rf = baseline_results['RandomForest']['roc_auc']

print(f"\nLogisticRegression ROC-AUC:")
print(f"  Baseline:         {baseline_auc_lr:.4f}")
print(f"  Class Weights:    {weighted_results['LogisticRegression_weighted']['roc_auc']:.4f} "
      f"(Δ {weighted_results['LogisticRegression_weighted']['roc_auc'] - baseline_auc_lr:+.4f})")
print(f"  SMOTE:            {smote_results['LogisticRegression_SMOTE']['roc_auc']:.4f} "
      f"(Δ {smote_results['LogisticRegression_SMOTE']['roc_auc'] - baseline_auc_lr:+.4f})")
print(f"  Undersampling:    {rus_results['LogisticRegression_RUS']['roc_auc']:.4f} "
      f"(Δ {rus_results['LogisticRegression_RUS']['roc_auc'] - baseline_auc_lr:+.4f})")

print(f"\nRandomForest ROC-AUC:")
print(f"  Baseline:         {baseline_auc_rf:.4f}")
print(f"  Class Weights:    {weighted_results['RandomForest_weighted']['roc_auc']:.4f} "
      f"(Δ {weighted_results['RandomForest_weighted']['roc_auc'] - baseline_auc_rf:+.4f})")
print(f"  SMOTE:            {smote_results['RandomForest_SMOTE']['roc_auc']:.4f} "
      f"(Δ {smote_results['RandomForest_SMOTE']['roc_auc'] - baseline_auc_rf:+.4f})")
print(f"  Undersampling:    {rus_results['RandomForest_RUS']['roc_auc']:.4f} "
      f"(Δ {rus_results['RandomForest_RUS']['roc_auc'] - baseline_auc_rf:+.4f})")

# F1-score (also sensitive to imbalance)
print("\n--- F1-Score Metric ---")
baseline_f1_lr = baseline_results['LogisticRegression']['f1']
baseline_f1_rf = baseline_results['RandomForest']['f1']

print(f"\nLogisticRegression F1-Score:")
print(f"  Baseline:         {baseline_f1_lr:.4f}")
print(f"  Class Weights:    {weighted_results['LogisticRegression_weighted']['f1']:.4f} "
      f"(Δ {weighted_results['LogisticRegression_weighted']['f1'] - baseline_f1_lr:+.4f})")
print(f"  SMOTE:            {smote_results['LogisticRegression_SMOTE']['f1']:.4f} "
      f"(Δ {smote_results['LogisticRegression_SMOTE']['f1'] - baseline_f1_lr:+.4f})")
print(f"  Undersampling:    {rus_results['LogisticRegression_RUS']['f1']:.4f} "
      f"(Δ {rus_results['LogisticRegression_RUS']['f1'] - baseline_f1_lr:+.4f})")

print(f"\nRandomForest F1-Score:")
print(f"  Baseline:         {baseline_f1_rf:.4f}")
print(f"  Class Weights:    {weighted_results['RandomForest_weighted']['f1']:.4f} "
      f"(Δ {weighted_results['RandomForest_weighted']['f1'] - baseline_f1_rf:+.4f})")
print(f"  SMOTE:            {smote_results['RandomForest_SMOTE']['f1']:.4f} "
      f"(Δ {smote_results['RandomForest_SMOTE']['f1'] - baseline_f1_rf:+.4f})")
print(f"  Undersampling:    {rus_results['RandomForest_RUS']['f1']:.4f} "
      f"(Δ {rus_results['RandomForest_RUS']['f1'] - baseline_f1_rf:+.4f})")

# Balanced Accuracy (metric designed for imbalanced data)
print("\n--- Balanced Accuracy Metric ---")
baseline_ba_lr = baseline_results['LogisticRegression']['balanced_accuracy']
baseline_ba_rf = baseline_results['RandomForest']['balanced_accuracy']

print(f"\nLogisticRegression Balanced Accuracy:")
print(f"  Baseline:         {baseline_ba_lr:.4f}")
print(f"  Class Weights:    {weighted_results['LogisticRegression_weighted']['balanced_accuracy']:.4f} "
      f"(Δ {weighted_results['LogisticRegression_weighted']['balanced_accuracy'] - baseline_ba_lr:+.4f})")
print(f"  SMOTE:            {smote_results['LogisticRegression_SMOTE']['balanced_accuracy']:.4f} "
      f"(Δ {smote_results['LogisticRegression_SMOTE']['balanced_accuracy'] - baseline_ba_lr:+.4f})")
print(f"  Undersampling:    {rus_results['LogisticRegression_RUS']['balanced_accuracy']:.4f} "
      f"(Δ {rus_results['LogisticRegression_RUS']['balanced_accuracy'] - baseline_ba_lr:+.4f})")

print(f"\nRandomForest Balanced Accuracy:")
print(f"  Baseline:         {baseline_ba_rf:.4f}")
print(f"  Class Weights:    {weighted_results['RandomForest_weighted']['balanced_accuracy']:.4f} "
      f"(Δ {weighted_results['RandomForest_weighted']['balanced_accuracy'] - baseline_ba_rf:+.4f})")
print(f"  SMOTE:            {smote_results['RandomForest_SMOTE']['balanced_accuracy']:.4f} "
      f"(Δ {smote_results['RandomForest_SMOTE']['balanced_accuracy'] - baseline_ba_rf:+.4f})")
print(f"  Undersampling:    {rus_results['RandomForest_RUS']['balanced_accuracy']:.4f} "
      f"(Δ {rus_results['RandomForest_RUS']['balanced_accuracy'] - baseline_ba_rf:+.4f})")

# Overall assessment
print("\n" + "=" * 70)
print("OVERALL FINDINGS")
print("=" * 70)

# Compute average improvements
improvements_weighted = []
improvements_smote = []
improvements_rus = []

for model_type in ['LogisticRegression', 'RandomForest']:
    baseline = baseline_results[model_type]['roc_auc']
    weighted = weighted_results[f'{model_type}_weighted']['roc_auc']
    smote_val = smote_results[f'{model_type}_SMOTE']['roc_auc']
    rus_val = rus_results[f'{model_type}_RUS']['roc_auc']

    improvements_weighted.append(weighted - baseline)
    improvements_smote.append(smote_val - baseline)
    improvements_rus.append(rus_val - baseline)

avg_improvement_weighted = np.mean(improvements_weighted)
avg_improvement_smote = np.mean(improvements_smote)
avg_improvement_rus = np.mean(improvements_rus)

print(f"\nAverage ROC-AUC improvement across models:")
print(f"  Class Weights:    {avg_improvement_weighted:+.4f}")
print(f"  SMOTE:            {avg_improvement_smote:+.4f}")
print(f"  Undersampling:    {avg_improvement_rus:+.4f}")

# Best strategy
best_strategy = max(
    [('Class Weights', avg_improvement_weighted),
     ('SMOTE', avg_improvement_smote),
     ('Undersampling', avg_improvement_rus)],
    key=lambda x: x[1]
)

print(f"\nBest imbalance handling strategy: {best_strategy[0]} "
      f"(avg ROC-AUC improvement: {best_strategy[1]:+.4f})")

# Did ANY imbalance handling improve performance?
any_improvement = (avg_improvement_weighted > 0 or avg_improvement_smote > 0 or avg_improvement_rus > 0)

print(f"\nConclusion: Addressing class imbalance "
      f"{'IMPROVED' if any_improvement else 'DID NOT IMPROVE'} model performance "
      f"on this dataset (using ROC-AUC as primary metric).")

# Summary statistics for result.json
print("\n" + "=" * 70)
print("SUMMARY METRICS")
print("=" * 70)
print(f"Baseline RF ROC-AUC: {baseline_auc_rf:.4f}")
print(f"Best imbalance-handled ROC-AUC: {max([
    weighted_results['RandomForest_weighted']['roc_auc'],
    smote_results['RandomForest_SMOTE']['roc_auc'],
    rus_results['RandomForest_RUS']['roc_auc']
]):.4f}")
print(f"Max improvement: {best_strategy[1]:+.4f}")
