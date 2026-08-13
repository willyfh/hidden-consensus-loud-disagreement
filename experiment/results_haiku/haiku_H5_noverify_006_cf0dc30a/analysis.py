import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import json
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nTarget class distribution:")
print(df['class'].value_counts())
print("\nTarget class proportions:")
print(df['class'].value_counts(normalize=True))

# Data preprocessing
# Create a copy for processing
data = df.copy()

# Separate features and target
X = data.drop('class', axis=1)
y = data['class']

# Convert target to binary (1 for >50K, 0 for <=50K)
y_binary = (y == '>50K').astype(int)

print(f"\nMinority class (>50K) proportion: {y_binary.mean():.4f}")
print(f"Minority class (>50K) count: {y_binary.sum()}")
print(f"Majority class (<=50K) count: {(1-y_binary).sum()}")

# Handle categorical and numerical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical variables
label_encoders = {}
X_encoded = X.copy()
for col in categorical_cols:
    # Handle missing values in categorical columns
    X_encoded[col] = X_encoded[col].fillna('Unknown')
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le

# Handle missing values in numerical columns
X_encoded = X_encoded.fillna(X_encoded.mean())

print(f"\nEncoded features shape: {X_encoded.shape}")

# Train-test split (stratified by target to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_binary, test_size=0.2, random_state=42, stratify=y_binary
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train set minority class proportion: {y_train.mean():.4f}")
print(f"Test set minority class proportion: {y_test.mean():.4f}")

# Model 1: Random Forest WITHOUT SMOTE
print("\n" + "="*60)
print("Model 1: Random Forest WITHOUT SMOTE")
print("="*60)

rf_no_smote = RandomForestClassifier(random_state=42)
rf_no_smote.fit(X_train, y_train)
y_pred_no_smote = rf_no_smote.predict(X_test)

# Calculate F1 score for minority class (pos_label=1)
f1_no_smote = f1_score(y_test, y_pred_no_smote, pos_label=1)
print(f"F1 score (minority class >50K): {f1_no_smote:.4f}")

# Model 2: Random Forest WITH SMOTE
print("\n" + "="*60)
print("Model 2: Random Forest WITH SMOTE")
print("="*60)

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Train set size: {X_train_smote.shape[0]}")
print(f"Train set minority class proportion: {y_train_smote.mean():.4f}")
print(f"Train set minority class count: {y_train_smote.sum()}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)

# Calculate F1 score for minority class (pos_label=1)
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)
print(f"F1 score (minority class >50K): {f1_smote:.4f}")

# Calculate the difference
f1_difference = f1_smote - f1_no_smote
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
print(f"F1 score WITHOUT SMOTE: {f1_no_smote:.4f}")
print(f"F1 score WITH SMOTE:    {f1_smote:.4f}")
print(f"Difference (SMOTE - No SMOTE): {f1_difference:.4f}")
print(f"Threshold (0.02): {0.02:.4f}")
print(f"Does difference exceed 0.02? {abs(f1_difference) > 0.02}")

# Determine if the difference exceeds 0.02
exceeds_threshold = abs(f1_difference) > 0.02
direction_str = f"SMOTE F1 {'higher' if f1_difference > 0 else 'lower'} by {abs(f1_difference):.4f}"

# Additional metrics for context
from sklearn.metrics import precision_score, recall_score, roc_auc_score

print("\n" + "="*60)
print("DETAILED METRICS - Without SMOTE")
print("="*60)
precision_no_smote = precision_score(y_test, y_pred_no_smote, pos_label=1)
recall_no_smote = recall_score(y_test, y_pred_no_smote, pos_label=1)
roc_auc_no_smote = roc_auc_score(y_test, rf_no_smote.predict_proba(X_test)[:, 1])
print(f"Precision (minority class): {precision_no_smote:.4f}")
print(f"Recall (minority class): {recall_no_smote:.4f}")
print(f"ROC-AUC: {roc_auc_no_smote:.4f}")

print("\n" + "="*60)
print("DETAILED METRICS - With SMOTE")
print("="*60)
precision_smote = precision_score(y_test, y_pred_smote, pos_label=1)
recall_smote = recall_score(y_test, y_pred_smote, pos_label=1)
roc_auc_smote = roc_auc_score(y_test, rf_smote.predict_proba(X_test)[:, 1])
print(f"Precision (minority class): {precision_smote:.4f}")
print(f"Recall (minority class): {recall_smote:.4f}")
print(f"ROC-AUC: {roc_auc_smote:.4f}")

# Prepare result
result = {
    "hypothesis_id": "H5",
    "summary": f"Applying SMOTE oversampling {'does' if exceeds_threshold else 'does not'} change the minority-class (>50K) F1 score by more than 0.02. The difference in F1 score is {f1_difference:.4f} (SMOTE: {f1_smote:.4f} vs. No SMOTE: {f1_no_smote:.4f}).",
    "primary_metric_name": "F1 score difference for minority class (>50K) - SMOTE vs. No SMOTE",
    "primary_metric_value": round(f1_difference, 6),
    "direction": direction_str,
    "methodological_choices": (
        "Train-test split: 80-20 stratified split to maintain class distribution. "
        "Preprocessing: Label encoding for categorical variables (workclass, education, marital-status, occupation, relationship, race, sex, native-country), "
        "mean imputation for missing numerical values. "
        "Target encoding: Binary (1 for >50K, 0 for <=50K). "
        "Classifier: RandomForestClassifier with default hyperparameters (n_estimators=100, random_state=42). "
        "SMOTE applied only to training data with random_state=42. "
        "Metric: F1 score for minority class (>50K) using pos_label=1. "
        "Evaluation: F1 scores calculated on held-out test set."
    )
}

# Save result to JSON
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*60)
print("RESULT SUMMARY")
print("="*60)
print(json.dumps(result, indent=2))
print("\nResults saved to result.json")
