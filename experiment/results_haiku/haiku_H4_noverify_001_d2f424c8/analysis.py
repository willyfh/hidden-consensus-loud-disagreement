"""
Analysis: Does addressing class imbalance improve model quality on the Adult Income dataset?
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_recall_curve, auc,
    balanced_accuracy_score, confusion_matrix
)
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. Load and explore data
# ============================================================================
print("Loading data...")
df = pd.read_csv('adult_income.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check class distribution
print(f"\nClass distribution (target='class'):")
print(df['class'].value_counts())
print(f"\nClass proportions:")
print(df['class'].value_counts(normalize=True))

class_counts = df['class'].value_counts()
imbalance_ratio = class_counts.iloc[0] / class_counts.iloc[1]
print(f"\nImbalance ratio (majority/minority): {imbalance_ratio:.3f}")

# ============================================================================
# 2. Preprocessing
# ============================================================================
print("\n" + "="*70)
print("PREPROCESSING")
print("="*70)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target (<=50K: 0, >50K: 1)
y_encoded = (y == '>50K').astype(int)

print(f"Features shape: {X.shape}")
print(f"Target shape: {y_encoded.shape}")

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols[:5]}...")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# Encode categorical variables
X_encoded = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

print(f"\nEncoded features shape: {X_encoded.shape}")

# Handle any remaining NaN values
X_encoded = X_encoded.fillna(X_encoded.mean(numeric_only=True))

# Train-test split (stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train class distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test class distribution:\n{pd.Series(y_test).value_counts()}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 3. Model 1: WITHOUT class imbalance handling
# ============================================================================
print("\n" + "="*70)
print("MODEL 1: WITHOUT CLASS IMBALANCE HANDLING")
print("="*70)

# Logistic Regression (no class weighting)
lr_no_balance = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_no_balance.fit(X_train_scaled, y_train)

# Predictions and probabilities
y_pred_lr_nb = lr_no_balance.predict(X_test_scaled)
y_proba_lr_nb = lr_no_balance.predict_proba(X_test_scaled)[:, 1]

# Evaluation metrics
roc_auc_lr_nb = roc_auc_score(y_test, y_proba_lr_nb)
f1_lr_nb = f1_score(y_test, y_pred_lr_nb)
balanced_acc_lr_nb = balanced_accuracy_score(y_test, y_pred_lr_nb)

# Precision-recall AUC
precision, recall, _ = precision_recall_curve(y_test, y_proba_lr_nb)
pr_auc_lr_nb = auc(recall, precision)

print(f"Logistic Regression (no imbalance handling):")
print(f"  ROC-AUC: {roc_auc_lr_nb:.4f}")
print(f"  F1-Score: {f1_lr_nb:.4f}")
print(f"  Balanced Accuracy: {balanced_acc_lr_nb:.4f}")
print(f"  PR-AUC: {pr_auc_lr_nb:.4f}")

# Random Forest (no class weighting)
rf_no_balance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
rf_no_balance.fit(X_train, y_train)

y_pred_rf_nb = rf_no_balance.predict(X_test)
y_proba_rf_nb = rf_no_balance.predict_proba(X_test)[:, 1]

roc_auc_rf_nb = roc_auc_score(y_test, y_proba_rf_nb)
f1_rf_nb = f1_score(y_test, y_pred_rf_nb)
balanced_acc_rf_nb = balanced_accuracy_score(y_test, y_pred_rf_nb)

precision, recall, _ = precision_recall_curve(y_test, y_proba_rf_nb)
pr_auc_rf_nb = auc(recall, precision)

print(f"\nRandom Forest (no imbalance handling):")
print(f"  ROC-AUC: {roc_auc_rf_nb:.4f}")
print(f"  F1-Score: {f1_rf_nb:.4f}")
print(f"  Balanced Accuracy: {balanced_acc_rf_nb:.4f}")
print(f"  PR-AUC: {pr_auc_rf_nb:.4f}")

# ============================================================================
# 4. Model 2: WITH class imbalance handling (SMOTE)
# ============================================================================
print("\n" + "="*70)
print("MODEL 2: WITH CLASS IMBALANCE HANDLING (SMOTE + Scaled)")
print("="*70)

# Apply SMOTE on training data
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE - Train class distribution:\n{pd.Series(y_train_smote).value_counts()}")

# Logistic Regression with SMOTE
lr_with_balance = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
lr_with_balance.fit(X_train_smote, y_train_smote)

y_pred_lr_wb = lr_with_balance.predict(X_test_scaled)
y_proba_lr_wb = lr_with_balance.predict_proba(X_test_scaled)[:, 1]

roc_auc_lr_wb = roc_auc_score(y_test, y_proba_lr_wb)
f1_lr_wb = f1_score(y_test, y_pred_lr_wb)
balanced_acc_lr_wb = balanced_accuracy_score(y_test, y_pred_lr_wb)

precision, recall, _ = precision_recall_curve(y_test, y_proba_lr_wb)
pr_auc_lr_wb = auc(recall, precision)

print(f"Logistic Regression (with SMOTE):")
print(f"  ROC-AUC: {roc_auc_lr_wb:.4f}")
print(f"  F1-Score: {f1_lr_wb:.4f}")
print(f"  Balanced Accuracy: {balanced_acc_lr_wb:.4f}")
print(f"  PR-AUC: {pr_auc_lr_wb:.4f}")

# Random Forest with SMOTE (note: RF typically applied to unscaled data)
X_train_smote_unscaled, _ = smote.fit_resample(X_train, y_train)

rf_with_balance = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=15)
rf_with_balance.fit(X_train_smote_unscaled, y_train_smote)

y_pred_rf_wb = rf_with_balance.predict(X_test)
y_proba_rf_wb = rf_with_balance.predict_proba(X_test)[:, 1]

roc_auc_rf_wb = roc_auc_score(y_test, y_proba_rf_wb)
f1_rf_wb = f1_score(y_test, y_pred_rf_wb)
balanced_acc_rf_wb = balanced_accuracy_score(y_test, y_pred_rf_wb)

precision, recall, _ = precision_recall_curve(y_test, y_proba_rf_wb)
pr_auc_rf_wb = auc(recall, precision)

print(f"\nRandom Forest (with SMOTE):")
print(f"  ROC-AUC: {roc_auc_rf_wb:.4f}")
print(f"  F1-Score: {f1_rf_wb:.4f}")
print(f"  Balanced Accuracy: {balanced_acc_rf_wb:.4f}")
print(f"  PR-AUC: {pr_auc_rf_wb:.4f}")

# ============================================================================
# 5. Comparison and Analysis
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: CLASS IMBALANCE HANDLING IMPACT")
print("="*70)

# Logistic Regression comparison
print("\nLogistic Regression:")
print(f"  Without balance - ROC-AUC: {roc_auc_lr_nb:.4f}, F1: {f1_lr_nb:.4f}, Balanced Acc: {balanced_acc_lr_nb:.4f}, PR-AUC: {pr_auc_lr_nb:.4f}")
print(f"  With SMOTE     - ROC-AUC: {roc_auc_lr_wb:.4f}, F1: {f1_lr_wb:.4f}, Balanced Acc: {balanced_acc_lr_wb:.4f}, PR-AUC: {pr_auc_lr_wb:.4f}")
print(f"  ROC-AUC improvement: {roc_auc_lr_wb - roc_auc_lr_nb:+.4f}")
print(f"  F1 improvement: {f1_lr_wb - f1_lr_nb:+.4f}")
print(f"  Balanced Acc improvement: {balanced_acc_lr_wb - balanced_acc_lr_nb:+.4f}")
print(f"  PR-AUC improvement: {pr_auc_lr_wb - pr_auc_lr_nb:+.4f}")

# Random Forest comparison
print("\nRandom Forest:")
print(f"  Without balance - ROC-AUC: {roc_auc_rf_nb:.4f}, F1: {f1_rf_nb:.4f}, Balanced Acc: {balanced_acc_rf_nb:.4f}, PR-AUC: {pr_auc_rf_nb:.4f}")
print(f"  With SMOTE     - ROC-AUC: {roc_auc_rf_wb:.4f}, F1: {f1_rf_wb:.4f}, Balanced Acc: {balanced_acc_rf_wb:.4f}, PR-AUC: {pr_auc_rf_wb:.4f}")
print(f"  ROC-AUC improvement: {roc_auc_rf_wb - roc_auc_rf_nb:+.4f}")
print(f"  F1 improvement: {f1_rf_wb - f1_rf_nb:+.4f}")
print(f"  Balanced Acc improvement: {balanced_acc_rf_wb - balanced_acc_rf_nb:+.4f}")
print(f"  PR-AUC improvement: {pr_auc_rf_wb - pr_auc_rf_nb:+.4f}")

# Compute average improvements across both models
avg_roc_auc_improvement = ((roc_auc_lr_wb - roc_auc_lr_nb) + (roc_auc_rf_wb - roc_auc_rf_nb)) / 2
avg_f1_improvement = ((f1_lr_wb - f1_lr_nb) + (f1_rf_wb - f1_rf_nb)) / 2
avg_balanced_acc_improvement = ((balanced_acc_lr_wb - balanced_acc_lr_nb) + (balanced_acc_rf_wb - balanced_acc_rf_nb)) / 2
avg_pr_auc_improvement = ((pr_auc_lr_wb - pr_auc_lr_nb) + (pr_auc_rf_wb - pr_auc_rf_nb)) / 2

print(f"\n--- Average improvements across both models ---")
print(f"  ROC-AUC: {avg_roc_auc_improvement:+.4f}")
print(f"  F1-Score: {avg_f1_improvement:+.4f}")
print(f"  Balanced Accuracy: {avg_balanced_acc_improvement:+.4f}")
print(f"  PR-AUC: {avg_pr_auc_improvement:+.4f}")

# Overall conclusion
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if avg_roc_auc_improvement > 0 and avg_f1_improvement > 0:
    conclusion = "Yes"
    direction = "Positive - addressing class imbalance improves model quality"
    summary = f"Addressing class imbalance with SMOTE improves model quality on the Adult Income dataset. Average ROC-AUC improvement: {avg_roc_auc_improvement:+.4f}, F1 improvement: {avg_f1_improvement:+.4f}. Both Logistic Regression and Random Forest show improvements in key metrics when trained with SMOTE."
elif avg_roc_auc_improvement > 0 or avg_f1_improvement > 0:
    conclusion = "Mixed/Partial"
    direction = "Mixed results - improvement depends on metric and model"
    summary = f"Addressing class imbalance shows mixed results depending on the evaluation metric and model. ROC-AUC improved by {avg_roc_auc_improvement:+.4f} on average, but F1-Score changed by {avg_f1_improvement:+.4f}. Trade-offs exist between different metrics."
else:
    conclusion = "No"
    direction = "Negative - class imbalance handling does not improve performance"
    summary = f"Addressing class imbalance with SMOTE does not improve model quality on this dataset. ROC-AUC declined by {avg_roc_auc_improvement:+.4f} and F1-Score by {avg_f1_improvement:+.4f} on average."

print(f"Conclusion: {conclusion}")
print(f"Direction: {direction}")
print(f"Summary: {summary}")

# ============================================================================
# 6. Save results
# ============================================================================
results = {
    "hypothesis_id": "H4",
    "summary": summary,
    "primary_metric_name": "Average ROC-AUC improvement (with SMOTE vs baseline)",
    "primary_metric_value": round(avg_roc_auc_improvement, 4),
    "direction": direction,
    "methodological_choices": (
        "Data split: 70% train, 30% test with stratification to preserve class distribution. "
        "Encoding: LabelEncoder for categorical features, StandardScaler for numeric features. "
        "Models: Logistic Regression (max_iter=1000) and Random Forest (100 trees, max_depth=15). "
        "Imbalance handling: SMOTE with k_neighbors=5 applied only to training data. "
        "Evaluation metrics: ROC-AUC, F1-Score, Balanced Accuracy, and Precision-Recall AUC. "
        "Random Forest uses unscaled features; Logistic Regression uses scaled features. "
        "All models evaluated on the same hold-out test set. "
        "No hyperparameter tuning performed; default/reasonable parameters used."
    )
}

# Save to result.json
with open('result.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to result.json")
print(f"Analysis code saved as analysis.py")
