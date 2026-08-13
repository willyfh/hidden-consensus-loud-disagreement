import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names and types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget class distribution:")
print(df['class'].value_counts())
print("Target class proportions:")
print(df['class'].value_counts(normalize=True))

# Data preprocessing
# Handle missing values (commonly represented as '?' or ' ?')
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].replace('?', np.nan)
        df[col] = df[col].str.strip() if df[col].dtype == 'object' else df[col]

# Drop rows with missing values
df_clean = df.dropna()
print(f"\nAfter dropping missing values: {df_clean.shape}")
print("Target class distribution after cleaning:")
print(df_clean['class'].value_counts())

# Separate features and target
X = df_clean.drop('class', axis=1)
y = df_clean['class']

# Encode target (minority class is >50K)
y_encoded = (y == '>50K').astype(int)
print(f"\nTarget encoding: 0 = <=50K, 1 = >50K")
print(f"Minority class (>50K) proportion: {y_encoded.mean():.4f}")

# Encode categorical features
X_encoded = X.copy()
label_encoders = {}
categorical_cols = X_encoded.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

print(f"\nEncoded {len(categorical_cols)} categorical columns")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nTrain set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
print(f"Train minority class proportion: {y_train.mean():.4f}")
print(f"Test minority class proportion: {y_test.mean():.4f}")

# Model 1: Random Forest without SMOTE
print("\n" + "="*60)
print("Model 1: Random Forest WITHOUT SMOTE")
print("="*60)

rf_no_smote = RandomForestClassifier(random_state=42)
rf_no_smote.fit(X_train, y_train)
y_pred_no_smote = rf_no_smote.predict(X_test)

f1_no_smote = f1_score(y_test, y_pred_no_smote, pos_label=1)
print(f"F1 score (minority class / >50K): {f1_no_smote:.6f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred_no_smote, target_names=['<=50K', '>50K']))

# Model 2: Random Forest with SMOTE
print("\n" + "="*60)
print("Model 2: Random Forest WITH SMOTE oversampling")
print("="*60)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - training set size: {X_train_smote.shape}")
print(f"After SMOTE - training minority class proportion: {y_train_smote.mean():.4f}")

rf_with_smote = RandomForestClassifier(random_state=42)
rf_with_smote.fit(X_train_smote, y_train_smote)
y_pred_with_smote = rf_with_smote.predict(X_test)

f1_with_smote = f1_score(y_test, y_pred_with_smote, pos_label=1)
print(f"F1 score (minority class / >50K): {f1_with_smote:.6f}")
print("\nClassification report:")
print(classification_report(y_test, y_pred_with_smote, target_names=['<=50K', '>50K']))

# Analysis of results
print("\n" + "="*60)
print("RESULTS SUMMARY")
print("="*60)
f1_difference = f1_with_smote - f1_no_smote
print(f"F1 score without SMOTE: {f1_no_smote:.6f}")
print(f"F1 score with SMOTE:    {f1_with_smote:.6f}")
print(f"Difference (SMOTE - No SMOTE): {f1_difference:.6f}")
print(f"\nAbsolute difference: {abs(f1_difference):.6f}")
print(f"Threshold for significance: 0.02")
print(f"Is difference > 0.02? {abs(f1_difference) > 0.02}")

# Research question answer
if abs(f1_difference) > 0.02:
    if f1_difference > 0:
        direction = f"SMOTE improves F1 score by {f1_difference:.6f}"
    else:
        direction = f"SMOTE decreases F1 score by {abs(f1_difference):.6f}"
    conclusion = f"YES - SMOTE changes the minority-class F1 score by {abs(f1_difference):.6f}, which is more than the 0.02 threshold."
else:
    direction = f"SMOTE changes F1 score by {f1_difference:.6f} (within 0.02 threshold)"
    conclusion = f"NO - SMOTE changes the minority-class F1 score by {abs(f1_difference):.6f}, which is NOT more than the 0.02 threshold."

print(f"\nConclusion: {conclusion}")

# Save results
result = {
    "hypothesis_id": "H5",
    "summary": conclusion,
    "primary_metric_name": "Minority-class (>50K) F1 score difference (SMOTE - No SMOTE)",
    "primary_metric_value": round(f1_difference, 6),
    "direction": direction,
    "methodological_choices": (
        "Train-test split: 70-30 stratified on target class. "
        "Preprocessing: removed rows with missing values, label-encoded categorical features. "
        "SMOTE applied only to training data before fitting the model. "
        "Model: RandomForestClassifier() with default hyperparameters. "
        "Evaluation metric: F1 score for minority class (>50K), calculated on held-out test set. "
        "Random state: 42 for reproducibility."
    )
}

with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*60)
print("Results saved to result.json")
print("="*60)
