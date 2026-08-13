import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import json
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget distribution:")
print(df['class'].value_counts())
print("\nTarget proportions:")
print(df['class'].value_counts(normalize=True))

# Data preprocessing
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

# Handle missing values (represented as '?')
df = df.replace('?', np.nan)
print("\nMissing values after replacing '?':")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Drop rows with missing values
df = df.dropna()
print(f"\nDataset shape after dropping NaNs: {df.shape}")

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

print(f"\nTarget distribution after cleaning:")
print(y.value_counts())
print("\nTarget proportions:")
print(y.value_counts(normalize=True))

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encode target variable
y_encoded = LabelEncoder().fit_transform(y)
# y_encoded: 0 = '<=50K', 1 = '>50K' (minority class)

print(f"\nEncoded target: 0='<=50K', 1='>50K'")
print(f"Minority class (1) proportion: {(y_encoded == 1).sum() / len(y_encoded):.4f}")

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Train target distribution: {np.bincount(y_train)}")
print(f"Test target distribution: {np.bincount(y_test)}")

# Condition 1: No resampling
print("\n" + "="*60)
print("MODEL 1: Random Forest WITHOUT SMOTE")
print("="*60)

rf_no_smote = RandomForestClassifier(random_state=42)
rf_no_smote.fit(X_train, y_train)
y_pred_no_smote = rf_no_smote.predict(X_test)

# F1 score for minority class (>50K, label=1)
f1_no_smote = f1_score(y_test, y_pred_no_smote, pos_label=1)
print(f"\nF1 Score (minority class, >50K): {f1_no_smote:.6f}")

# Additional metrics
from sklearn.metrics import precision_score, recall_score, confusion_matrix
precision_no_smote = precision_score(y_test, y_pred_no_smote, pos_label=1)
recall_no_smote = recall_score(y_test, y_pred_no_smote, pos_label=1)
cm_no_smote = confusion_matrix(y_test, y_pred_no_smote)

print(f"Precision (minority class): {precision_no_smote:.6f}")
print(f"Recall (minority class): {recall_no_smote:.6f}")
print(f"Confusion Matrix:\n{cm_no_smote}")

# Condition 2: With SMOTE
print("\n" + "="*60)
print("MODEL 2: Random Forest WITH SMOTE")
print("="*60)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE resampling:")
print(f"Train set size: {X_train_smote.shape[0]} samples")
print(f"Train target distribution: {np.bincount(y_train_smote)}")

rf_with_smote = RandomForestClassifier(random_state=42)
rf_with_smote.fit(X_train_smote, y_train_smote)
y_pred_with_smote = rf_with_smote.predict(X_test)

# F1 score for minority class (>50K, label=1)
f1_with_smote = f1_score(y_test, y_pred_with_smote, pos_label=1)
print(f"\nF1 Score (minority class, >50K): {f1_with_smote:.6f}")

precision_with_smote = precision_score(y_test, y_pred_with_smote, pos_label=1)
recall_with_smote = recall_score(y_test, y_pred_with_smote, pos_label=1)
cm_with_smote = confusion_matrix(y_test, y_pred_with_smote)

print(f"Precision (minority class): {precision_with_smote:.6f}")
print(f"Recall (minority class): {recall_with_smote:.6f}")
print(f"Confusion Matrix:\n{cm_with_smote}")

# Calculate the difference
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

f1_difference = f1_with_smote - f1_no_smote
print(f"\nF1 score WITHOUT SMOTE: {f1_no_smote:.6f}")
print(f"F1 score WITH SMOTE:    {f1_with_smote:.6f}")
print(f"Difference (SMOTE - No SMOTE): {f1_difference:.6f}")
print(f"Absolute difference: {abs(f1_difference):.6f}")
print(f"\nThreshold for 'significant change': 0.02")
print(f"Difference exceeds threshold: {abs(f1_difference) > 0.02}")

# Prepare result
result = {
    "hypothesis_id": "H5",
    "summary": f"SMOTE oversampling {'changed' if abs(f1_difference) > 0.02 else 'did not change'} the minority-class (>50K) F1 score by {abs(f1_difference):.4f} (difference: {f1_difference:+.4f}), which {'exceeds' if abs(f1_difference) > 0.02 else 'does not exceed'} the 0.02 threshold.",
    "primary_metric_name": "F1 score difference (SMOTE - No SMOTE)",
    "primary_metric_value": round(f1_difference, 6),
    "direction": f"SMOTE {'increased' if f1_difference > 0 else 'decreased' if f1_difference < 0 else 'had no effect on'} F1 by {abs(f1_difference):.4f}",
    "methodological_choices": (
        "Random Forest with default hyperparameters (random_state=42). "
        "Preprocessing: dropped rows with missing values, encoded categorical features with LabelEncoder. "
        "Train-test split: 70-30 with stratification by target. "
        "SMOTE applied to training data only to avoid data leakage. "
        "Evaluation: F1 score for minority class (>50K) on held-out test set. "
        "Threshold for significance: absolute difference > 0.02."
    )
}

print("\n" + "="*60)
print("RESULT")
print("="*60)
print(json.dumps(result, indent=2))

# Save result
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResult saved to result.json")
