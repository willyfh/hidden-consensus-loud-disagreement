import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE
import json

# Load data
df = pd.read_csv('adult_income.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nClass distribution:")
print(df['class'].value_counts())
print(f"\nMissing values per column:")
print(df.isnull().sum())

# Preprocess the data
# Remove rows with missing target
df = df.dropna(subset=['class'])

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Convert target to binary (1 for >50K, 0 for <=50K)
y_binary = (y == '>50K').astype(int)

print(f"\nBinary class distribution:")
print(f"Class 0 (<=50K): {(y_binary == 0).sum()}")
print(f"Class 1 (>50K): {(y_binary == 1).sum()}")

# Handle missing values in features
# For categorical columns, fill with mode
# For numerical columns, fill with median
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

for col in categorical_cols:
    X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown', inplace=True)

for col in numerical_cols:
    X[col].fillna(X[col].median(), inplace=True)

# Encode categorical features
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

print(f"\nFeature matrix shape after preprocessing: {X.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_binary, test_size=0.3, random_state=42, stratify=y_binary
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Model 1: RandomForest without SMOTE
print("\n" + "="*60)
print("Model 1: RandomForest WITHOUT SMOTE")
print("="*60)

rf_no_smote = RandomForestClassifier(random_state=42)
rf_no_smote.fit(X_train, y_train)

y_pred_no_smote = rf_no_smote.predict(X_test)
f1_no_smote = f1_score(y_test, y_pred_no_smote, pos_label=1)

print(f"F1 score (>50K class, minority): {f1_no_smote:.6f}")

# Model 2: RandomForest with SMOTE
print("\n" + "="*60)
print("Model 2: RandomForest WITH SMOTE on training data")
print("="*60)

# Apply SMOTE only to training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set size after SMOTE: {X_train_smote.shape[0]}")
print(f"Class distribution after SMOTE:")
print(f"  Class 0 (<=50K): {(y_train_smote == 0).sum()}")
print(f"  Class 1 (>50K): {(y_train_smote == 1).sum()}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = rf_smote.predict(X_test)
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)

print(f"F1 score (>50K class, minority): {f1_smote:.6f}")

# Calculate difference
f1_difference = f1_smote - f1_no_smote

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"F1 score without SMOTE: {f1_no_smote:.6f}")
print(f"F1 score with SMOTE:    {f1_smote:.6f}")
print(f"Difference (SMOTE - No SMOTE): {f1_difference:.6f}")
print(f"\nIs difference > 0.02? {abs(f1_difference) > 0.02}")

# Determine direction
if f1_difference > 0.02:
    direction = "SMOTE increases F1 by >0.02"
elif f1_difference < -0.02:
    direction = "SMOTE decreases F1 by >0.02"
else:
    direction = "No meaningful difference (|diff| ≤ 0.02)"

print(f"Direction: {direction}")

# Prepare result JSON
result = {
    "hypothesis_id": "H5",
    "summary": f"Applying SMOTE oversampling to the training data {'increases' if f1_difference > 0 else 'decreases'} the minority-class F1 score by {abs(f1_difference):.6f}. The difference {'is' if abs(f1_difference) > 0.02 else 'is not'} greater than 0.02.",
    "primary_metric_name": "Minority-class (>50K) F1 score difference (with SMOTE - without SMOTE)",
    "primary_metric_value": round(f1_difference, 6),
    "direction": direction,
    "methodological_choices": "Train-test split: 70-30 stratified split with random_state=42. Missing value handling: categorical features filled with mode, numerical features filled with median. Categorical encoding: LabelEncoder for all categorical features. SMOTE: Applied only to training data with random_state=42. Model: Default RandomForestClassifier() with random_state=42. Evaluation: F1 score (pos_label=1) on held-out test set."
}

# Save result
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResult saved to result.json")
