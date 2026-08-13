import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import json

# Load the data
print("Loading data...")
df = pd.read_csv('adult_income.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nTarget class distribution:")
print(df['class'].value_counts())
print(f"Class proportions:\n{df['class'].value_counts(normalize=True)}")

# Data preprocessing
print("\n" + "="*80)
print("PREPROCESSING")
print("="*80)

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Remove the target variable from features
if 'class' in categorical_cols:
    categorical_cols.remove('class')
if 'class' in numerical_cols:
    numerical_cols.remove('class')

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Handle missing values - replace '?' with NaN and drop
df_clean = df.copy()
df_clean = df_clean.replace('?', np.nan)
print(f"\nMissing values after ? replacement:")
print(df_clean.isnull().sum()[df_clean.isnull().sum() > 0])

# Drop rows with missing values
df_clean = df_clean.dropna()
print(f"Dataset shape after removing NaN: {df_clean.shape}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
    label_encoders[col] = le

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(df_clean['class'])
X = df_clean.drop('class', axis=1)

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target classes: {np.unique(y)} -> {le_target.classes_}")
print(f"Minority class (>50K) label: {np.where(le_target.classes_ == '>50K')[0][0]}")

# Train-test split (80-20)
print("\n" + "="*80)
print("TRAIN-TEST SPLIT")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set class distribution: {np.bincount(y_train)}")
print(f"Test set class distribution: {np.bincount(y_test)}")

minority_class_label = np.where(le_target.classes_ == '>50K')[0][0]
print(f"\nMinority class label in data: {minority_class_label}")

# Train RF without SMOTE
print("\n" + "="*80)
print("RANDOM FOREST WITHOUT SMOTE")
print("="*80)

rf_no_smote = RandomForestClassifier(random_state=42)
rf_no_smote.fit(X_train, y_train)

y_pred_no_smote = rf_no_smote.predict(X_test)
f1_no_smote = f1_score(y_test, y_pred_no_smote, pos_label=minority_class_label, zero_division=0)

print(f"F1 Score (minority class): {f1_no_smote:.6f}")
print(f"\nClassification Report (No SMOTE):")
print(classification_report(y_test, y_pred_no_smote, target_names=le_target.classes_))
print(f"Confusion Matrix (No SMOTE):")
print(confusion_matrix(y_test, y_pred_no_smote))

# Train RF with SMOTE
print("\n" + "="*80)
print("RANDOM FOREST WITH SMOTE")
print("="*80)

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set shape after SMOTE: {X_train_smote.shape}")
print(f"Training set class distribution after SMOTE: {np.bincount(y_train_smote)}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = rf_smote.predict(X_test)
f1_smote = f1_score(y_test, y_pred_smote, pos_label=minority_class_label, zero_division=0)

print(f"F1 Score (minority class): {f1_smote:.6f}")
print(f"\nClassification Report (With SMOTE):")
print(classification_report(y_test, y_pred_smote, target_names=le_target.classes_))
print(f"Confusion Matrix (With SMOTE):")
print(confusion_matrix(y_test, y_pred_smote))

# Compare results
print("\n" + "="*80)
print("COMPARISON")
print("="*80)

f1_difference = f1_smote - f1_no_smote
print(f"F1 Score without SMOTE: {f1_no_smote:.6f}")
print(f"F1 Score with SMOTE: {f1_smote:.6f}")
print(f"Difference (SMOTE - No SMOTE): {f1_difference:.6f}")
print(f"Threshold: 0.02")
print(f"Difference exceeds threshold: {abs(f1_difference) > 0.02}")

# Determine direction
if f1_difference > 0.02:
    direction = "SMOTE improves F1 by more than 0.02"
elif f1_difference < -0.02:
    direction = "SMOTE reduces F1 by more than 0.02"
else:
    direction = f"Difference of {f1_difference:.6f} does not exceed 0.02 threshold"

print(f"\nConclusion: {direction}")

# Prepare result JSON
result = {
    "hypothesis_id": "H5",
    "summary": f"SMOTE oversampling {'does' if abs(f1_difference) > 0.02 else 'does not'} change the minority-class F1 score by more than 0.02. The F1 score {'increased' if f1_difference > 0 else 'decreased'} from {f1_no_smote:.6f} (no resampling) to {f1_smote:.6f} (with SMOTE), a difference of {f1_difference:.6f}.",
    "primary_metric_name": "Minority-class F1 score difference (SMOTE - No SMOTE)",
    "primary_metric_value": round(f1_difference, 6),
    "direction": direction,
    "methodological_choices": "Preprocessed data by replacing '?' with NaN and removing missing values. Encoded categorical variables using LabelEncoder. Split data into 80/20 train/test with stratification. Applied SMOTE only on the training data before training. Used default RandomForestClassifier() with random_state=42 for reproducibility. Evaluated using F1 score for the minority class (>50K) on the held-out test set."
}

# Save result
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*80)
print("RESULT")
print("="*80)
print(json.dumps(result, indent=2))
