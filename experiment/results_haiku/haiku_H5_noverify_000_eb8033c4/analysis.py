import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import json
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
print("Loading data...")
df = pd.read_csv('adult_income.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumn names:\n{df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check target variable
print(f"\nTarget variable distribution:")
print(df['class'].value_counts())
print(f"\nTarget variable proportions:")
print(df['class'].value_counts(normalize=True))

# Identify minority class
class_counts = df['class'].value_counts()
minority_class = class_counts.idxmin()
print(f"\nMinority class: {minority_class}")

# ===== PREPROCESSING =====
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Handle missing values (represented as '?' or spaces)
print("\nHandling missing values...")
for col in X.columns:
    if X[col].dtype == 'object':
        # Replace '?' and empty strings with NaN
        X[col] = X[col].replace('?', np.nan)
        X[col] = X[col].replace(' ?', np.nan)
        # Fill with mode
        if X[col].isnull().sum() > 0:
            print(f"  {col}: {X[col].isnull().sum()} missing values -> filling with mode")
            X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else X[col].iloc[0])

# Encode categorical variables
print("\nEncoding categorical variables...")
categorical_cols = X.select_dtypes(include='object').columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le
    print(f"  {col}: encoded")

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)
print(f"\nTarget encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Train-test split (stratified to maintain class distribution)
print("\nTrain-test split (80-20, stratified)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set class distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test set class distribution:\n{pd.Series(y_test).value_counts()}")

# ===== BASELINE MODEL (NO RESAMPLING) =====
print("\n" + "="*60)
print("BASELINE: RandomForest WITHOUT SMOTE")
print("="*60)

rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train, y_train)

y_pred_baseline = rf_baseline.predict(X_test)

# Get F1 score for minority class
f1_baseline = f1_score(y_test, y_pred_baseline, pos_label=1)  # Assuming minority class is 1

print(f"\nBaseline model - F1 score (minority class): {f1_baseline:.4f}")
print(f"\nClassification Report (Baseline):")
print(classification_report(y_test, y_pred_baseline, digits=4))

# ===== MODEL WITH SMOTE =====
print("\n" + "="*60)
print("SMOTE MODEL: RandomForest WITH SMOTE")
print("="*60)

print("Applying SMOTE to training data...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set after SMOTE:")
print(f"  Original shape: {X_train.shape}")
print(f"  SMOTE shape: {X_train_smote.shape}")
print(f"  Class distribution: {pd.Series(y_train_smote).value_counts().to_dict()}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = rf_smote.predict(X_test)

# Get F1 score for minority class
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)

print(f"\nSMOTE model - F1 score (minority class): {f1_smote:.4f}")
print(f"\nClassification Report (SMOTE):")
print(classification_report(y_test, y_pred_smote, digits=4))

# ===== COMPARISON =====
print("\n" + "="*60)
print("COMPARISON & RESULTS")
print("="*60)

f1_difference = f1_smote - f1_baseline
print(f"\nF1 Score Baseline (No SMOTE): {f1_baseline:.4f}")
print(f"F1 Score SMOTE:              {f1_smote:.4f}")
print(f"Difference (SMOTE - Baseline): {f1_difference:.4f}")
print(f"Absolute Difference:           {abs(f1_difference):.4f}")

threshold = 0.02
exceeds_threshold = abs(f1_difference) > threshold
print(f"\nThreshold: {threshold}")
print(f"Does difference exceed threshold? {exceeds_threshold}")

# ===== OUTPUT RESULTS =====
summary = f"Applying SMOTE oversampling {'increases' if f1_difference > 0 else 'decreases'} the minority-class (>50K) F1 score from {f1_baseline:.4f} to {f1_smote:.4f}, a difference of {f1_difference:.4f}. This difference {'exceeds' if exceeds_threshold else 'does not exceed'} the 0.02 threshold."

result = {
    "hypothesis_id": "H5",
    "summary": summary,
    "primary_metric_name": "F1 Score difference (minority class, SMOTE - Baseline)",
    "primary_metric_value": round(f1_difference, 4),
    "direction": f"SMOTE {'improves' if f1_difference > 0 else 'degrades'} F1 by {abs(f1_difference):.4f}" + (f" (exceeds 0.02 threshold)" if exceeds_threshold else " (within 0.02 threshold)"),
    "methodological_choices": (
        "Train-test split: 80-20 stratified split (random_state=42). "
        "Preprocessing: Missing values (represented as '?') filled with mode for categorical features. "
        "All categorical variables label-encoded. "
        "Classifier: Default RandomForestClassifier() with no hyperparameter tuning. "
        "Resampling: SMOTE applied only to training data using default parameters. "
        "Evaluation: F1 score computed on held-out test set for minority class (pos_label=1, which is '>50K'). "
        "Minority class defined as the less frequent class in the dataset."
    )
}

print("\n" + "="*60)
print("FINAL RESULT")
print("="*60)
print(json.dumps(result, indent=2))

# Save results
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResults saved to result.json")
