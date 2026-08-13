import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import json

# Load the dataset
df = pd.read_csv('adult_income.csv')

# Preprocessing
print("Starting preprocessing...")

# Handle missing values by filling with mode for categorical features
df['workclass'] = df['workclass'].fillna(df['workclass'].mode()[0])
df['occupation'] = df['occupation'].fillna(df['occupation'].mode()[0])
df['native-country'] = df['native-country'].fillna(df['native-country'].mode()[0])

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"Target classes: {le_target.classes_}")

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Class distribution in train: {np.bincount(y_train)}")
print(f"Class distribution in test: {np.bincount(y_test)}")

# Train RF without resampling
print("\n--- Model 1: Random Forest without SMOTE ---")
rf_no_resample = RandomForestClassifier(random_state=42)
rf_no_resample.fit(X_train, y_train)
y_pred_no_resample = rf_no_resample.predict(X_test)

# Calculate F1 score for minority class (>50K, which is class 1)
f1_no_resample = f1_score(y_test, y_pred_no_resample, pos_label=1)
print(f"F1 score (minority class): {f1_no_resample:.6f}")
print(f"Classification report:")
print(classification_report(y_test, y_pred_no_resample, target_names=['<=50K', '>50K']))

# Train RF with SMOTE resampling
print("\n--- Model 2: Random Forest with SMOTE ---")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set size after SMOTE: {len(X_train_smote)}")
print(f"Class distribution in train after SMOTE: {np.bincount(y_train_smote)}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)

# Calculate F1 score for minority class
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)
print(f"F1 score (minority class): {f1_smote:.6f}")
print(f"Classification report:")
print(classification_report(y_test, y_pred_smote, target_names=['<=50K', '>50K']))

# Compare the results
print("\n--- Comparison ---")
f1_difference = f1_smote - f1_no_resample
print(f"F1 score without SMOTE: {f1_no_resample:.6f}")
print(f"F1 score with SMOTE: {f1_smote:.6f}")
print(f"Difference (SMOTE - No SMOTE): {f1_difference:.6f}")
print(f"Is difference > 0.02? {f1_difference > 0.02}")

# Prepare result
result = {
    "hypothesis_id": "H5",
    "summary": f"SMOTE oversampling {'increased' if f1_difference > 0 else 'decreased'} the minority-class (>50K) F1 score by {abs(f1_difference):.6f}, which is {'greater' if abs(f1_difference) > 0.02 else 'not greater'} than the 0.02 threshold.",
    "primary_metric_name": "F1 score difference (SMOTE - No SMOTE) for minority class (>50K)",
    "primary_metric_value": round(f1_difference, 6),
    "direction": f"SMOTE {'improves' if f1_difference > 0 else 'worsens'} F1 by {abs(f1_difference):.4f}" + (f" (exceeds 0.02 threshold)" if abs(f1_difference) > 0.02 else f" (within 0.02 threshold)"),
    "methodological_choices": (
        "Data preprocessing: filled missing values in workclass, occupation, and native-country with mode values. "
        "Feature encoding: LabelEncoder applied to all categorical features. "
        "Train-test split: 80-20 stratified split (random_state=42). "
        "Model: RandomForestClassifier with default hyperparameters (no tuning). "
        "SMOTE configuration: default settings (random_state=42). "
        "Evaluation metric: F1 score for minority class (>50K), calculated on held-out test set. "
        "The test set was not resampled, ensuring a fair evaluation of both models on the same data distribution."
    )
}

# Save result to JSON file
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n--- Result saved to result.json ---")
print(json.dumps(result, indent=2))
