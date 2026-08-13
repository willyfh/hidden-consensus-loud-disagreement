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

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target variable: >50K = 1, <=50K = 0
y_encoded = (y == '>50K').astype(int)

# Handle missing values
# For categorical columns, fill with mode; for numeric, fill with median
categorical_cols = X.select_dtypes(include='object').columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')

for col in numeric_cols:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Encode categorical variables
categorical_cols = X.select_dtypes(include='object').columns
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print("Dataset shapes:")
print(f"X_train: {X_train.shape}, y_train distribution: {np.bincount(y_train)}")
print(f"X_test: {X_test.shape}, y_test distribution: {np.bincount(y_test)}")

# ============================================================================
# Approach 1: Random Forest WITHOUT resampling
# ============================================================================
print("\n" + "="*70)
print("APPROACH 1: RandomForest WITHOUT resampling")
print("="*70)

rf_no_smote = RandomForestClassifier(random_state=42)
rf_no_smote.fit(X_train, y_train)

y_pred_no_smote = rf_no_smote.predict(X_test)
f1_no_smote = f1_score(y_test, y_pred_no_smote, pos_label=1)

print(f"\nF1 Score (minority class >50K): {f1_no_smote:.4f}")
print("\nClassification Report (No SMOTE):")
print(classification_report(y_test, y_pred_no_smote, target_names=['<=50K', '>50K']))

# ============================================================================
# Approach 2: Random Forest WITH SMOTE oversampling
# ============================================================================
print("\n" + "="*70)
print("APPROACH 2: RandomForest WITH SMOTE oversampling")
print("="*70)

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE, training set distribution: {np.bincount(y_train_smote)}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = rf_smote.predict(X_test)
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)

print(f"\nF1 Score (minority class >50K): {f1_smote:.4f}")
print("\nClassification Report (With SMOTE):")
print(classification_report(y_test, y_pred_smote, target_names=['<=50K', '>50K']))

# ============================================================================
# Comparison
# ============================================================================
print("\n" + "="*70)
print("COMPARISON")
print("="*70)

f1_difference = f1_smote - f1_no_smote
print(f"\nF1 Score (No SMOTE):     {f1_no_smote:.6f}")
print(f"F1 Score (With SMOTE):   {f1_smote:.6f}")
print(f"Difference (SMOTE - No): {f1_difference:.6f}")
print(f"\nAbsolute difference:     {abs(f1_difference):.6f}")
print(f"Is difference > 0.02?:   {abs(f1_difference) > 0.02}")

# ============================================================================
# Results for output
# ============================================================================
direction = (
    "SMOTE improves F1" if f1_smote > f1_no_smote
    else "SMOTE decreases F1" if f1_smote < f1_no_smote
    else "No difference"
)

if abs(f1_difference) > 0.02:
    result = f"Yes, SMOTE changes F1 by {abs(f1_difference):.4f} (> 0.02 threshold)"
else:
    result = f"No, SMOTE changes F1 by {abs(f1_difference):.4f} (≤ 0.02 threshold)"

print(f"\n\nFinal Answer: {result}")
print(f"Direction: {direction}")

# Save results to JSON
results = {
    "hypothesis_id": "H5",
    "summary": f"Applying SMOTE oversampling to training data {'changes' if abs(f1_difference) > 0.02 else 'does not substantially change'} the minority-class (>50K) F1 score by {abs(f1_difference):.4f} compared to no resampling (threshold: 0.02). The F1 score {'improved' if f1_smote > f1_no_smote else 'declined' if f1_smote < f1_no_smote else 'remained unchanged'} from {f1_no_smote:.4f} to {f1_smote:.4f}.",
    "primary_metric_name": "Minority-class (>50K) F1 score difference (SMOTE - No SMOTE)",
    "primary_metric_value": round(f1_difference, 6),
    "direction": direction,
    "methodological_choices": (
        "Data preprocessing: Missing values in categorical columns filled with mode, "
        "missing numeric values filled with median. "
        "Categorical encoding: LabelEncoder for all categorical features. "
        "Train/test split: 80/20 stratified split with random_state=42. "
        "Classifier: RandomForestClassifier with default hyperparameters (random_state=42). "
        "SMOTE configuration: k_neighbors=5, random_state=42 applied only to training set. "
        "Evaluation: F1 score calculated on held-out test set (pos_label=1 for >50K class). "
        "Threshold for 'substantial change': |F1_difference| > 0.02."
    )
}

with open('result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n\nResults saved to result.json")
