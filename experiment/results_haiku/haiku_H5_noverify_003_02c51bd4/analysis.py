"""
Analysis: Does SMOTE oversampling change minority-class F1 score by >0.02?

Research question: Does applying SMOTE oversampling to the training data change the
minority-class (>50K) F1 score by more than 0.02 compared to no resampling,
holding the classifier fixed as a default-hyperparameter random forest?
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import json
import warnings

warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:\n{df['class'].value_counts()}")
print(f"Class proportions:\n{df['class'].value_counts(normalize=True)}")

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Data preprocessing
print("\n" + "="*60)
print("DATA PREPROCESSING")
print("="*60)

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Handle missing values represented as '?'
X_clean = X.copy()
for col in categorical_cols:
    X_clean[col] = X_clean[col].replace('?', np.nan)

# Remove rows with missing values
print(f"\nRows before dropping missing: {len(X_clean)}")
mask = X_clean.isnull().any(axis=1)
X_clean = X_clean[~mask]
y_clean = y[~mask]
print(f"Rows after dropping missing: {len(X_clean)}")

# Encode categorical variables
le_dict = {}
X_encoded = X_clean.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    le_dict[col] = le

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y_clean)
print(f"\nTarget classes: {le_target.classes_}")
print(f"Minority class (>50K) is encoded as: {le_target.transform(['>50K'])[0]}")

# Train-test split
print("\n" + "="*60)
print("TRAIN-TEST SPLIT")
print("="*60)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Train class distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test class distribution:\n{pd.Series(y_test).value_counts()}")

# Identify minority class (>50K)
minority_class = le_target.transform(['>50K'])[0]
print(f"\nMinority class label: {minority_class}")

# Model 1: Baseline Random Forest (no resampling)
print("\n" + "="*60)
print("MODEL 1: BASELINE RANDOM FOREST (NO RESAMPLING)")
print("="*60)

rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train, y_train)
y_pred_baseline = rf_baseline.predict(X_test)

# Calculate F1 score for minority class
f1_baseline = f1_score(y_test, y_pred_baseline, pos_label=minority_class)
print(f"Minority-class F1 score (baseline): {f1_baseline:.6f}")
print(f"\nClassification Report (Baseline):")
print(classification_report(y_test, y_pred_baseline, target_names=le_target.classes_))

# Model 2: Random Forest with SMOTE
print("\n" + "="*60)
print("MODEL 2: RANDOM FOREST WITH SMOTE OVERSAMPLING")
print("="*60)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set size after SMOTE: {len(X_train_smote)}")
print(f"Train class distribution after SMOTE:\n{pd.Series(y_train_smote).value_counts()}")

# Train Random Forest on SMOTE-resampled data
rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)

# Calculate F1 score for minority class
f1_smote = f1_score(y_test, y_pred_smote, pos_label=minority_class)
print(f"Minority-class F1 score (SMOTE): {f1_smote:.6f}")
print(f"\nClassification Report (SMOTE):")
print(classification_report(y_test, y_pred_smote, target_names=le_target.classes_))

# Calculate difference
print("\n" + "="*60)
print("RESULTS")
print("="*60)

f1_difference = f1_smote - f1_baseline
threshold = 0.02

print(f"F1 score (baseline): {f1_baseline:.6f}")
print(f"F1 score (SMOTE):    {f1_smote:.6f}")
print(f"Difference (SMOTE - baseline): {f1_difference:.6f}")
print(f"Threshold: {threshold}")
print(f"\nDoes SMOTE change F1 by more than {threshold}?")
print(f"  |difference| = {abs(f1_difference):.6f}")
print(f"  Absolute difference > {threshold}? {abs(f1_difference) > threshold}")

if abs(f1_difference) > threshold:
    if f1_difference > 0:
        answer = f"YES - SMOTE improves F1 by {f1_difference:.6f}"
    else:
        answer = f"YES - SMOTE decreases F1 by {abs(f1_difference):.6f}"
else:
    answer = f"NO - Change is {abs(f1_difference):.6f}, which is <= {threshold}"

print(f"\nConclusion: {answer}")

# Save results to JSON
results = {
    "hypothesis_id": "H5",
    "summary": f"SMOTE oversampling changed the minority-class (>50K) F1 score by {f1_difference:.6f} (from {f1_baseline:.6f} to {f1_smote:.6f}), which is {'more' if abs(f1_difference) > threshold else 'less'} than the threshold of {threshold}.",
    "primary_metric_name": "Minority-class F1 score difference (SMOTE - Baseline)",
    "primary_metric_value": round(f1_difference, 6),
    "direction": f"SMOTE {'improves' if f1_difference > 0 else 'decreases'} F1 by {abs(f1_difference):.6f}" if abs(f1_difference) > threshold else f"No meaningful change (difference {f1_difference:.6f})",
    "methodological_choices": "Preprocessing: Removed rows with missing values (marked as '?'). Encoded all categorical variables using LabelEncoder. Train-test split: 80-20 stratified split with random_state=42. Model: RandomForestClassifier with default hyperparameters. SMOTE: Applied to training data only (not test data) with random_state=42. Evaluation: F1 score for minority class (>50K) calculated on held-out test set. Threshold for 'meaningful change': 0.02."
}

with open('result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Results saved to result.json")
print("="*60)
