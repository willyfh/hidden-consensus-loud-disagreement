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
print("\nTarget class distribution:")
print(df['class'].value_counts())
print("\nClass proportions:")
print(df['class'].value_counts(normalize=True))

# Preprocessing
# Handle missing values (represented as spaces or '?')
df = df.replace(' ?', np.nan)
df = df.replace('?', np.nan)

# Drop rows with missing values (simpler approach)
print(f"\nRows before dropping NAs: {len(df)}")
df = df.dropna()
print(f"Rows after dropping NAs: {len(df)}")

# Separate features and target
X = df.drop('class', axis=1)
y = df['class'].map({'<=50K': 0, '>50K': 1})

print(f"\nTarget distribution after preprocessing:")
print(f"Class 0 (<=50K): {(y==0).sum()}")
print(f"Class 1 (>50K): {(y==1).sum()}")
print(f"Minority class proportion: {(y==1).sum() / len(y):.4f}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Encode categorical variables
label_encoders = {}
X_processed = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

print(f"\nProcessed features shape: {X_processed.shape}")

# Train-test split (stratified to maintain class distribution)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"Train set - Class 1: {(y_train==1).sum()}, Class 0: {(y_train==0).sum()}")
print(f"Test set - Class 1: {(y_test==1).sum()}, Class 0: {(y_test==0).sum()}")

# ============================================
# Scenario 1: No resampling (baseline)
# ============================================
print("\n" + "="*60)
print("SCENARIO 1: No Resampling (Baseline)")
print("="*60)

rf_baseline = RandomForestClassifier(random_state=42)
rf_baseline.fit(X_train, y_train)
y_pred_baseline = rf_baseline.predict(X_test)

# Calculate F1 score for minority class (>50K, which is class 1)
f1_baseline = f1_score(y_test, y_pred_baseline, pos_label=1)
print(f"\nF1 Score (minority class >50K): {f1_baseline:.6f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_baseline, target_names=['<=50K', '>50K']))

# ============================================
# Scenario 2: With SMOTE oversampling
# ============================================
print("\n" + "="*60)
print("SCENARIO 2: With SMOTE Oversampling")
print("="*60)

# Apply SMOTE to training data only
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nTrain set after SMOTE:")
print(f"  Shape: {X_train_smote.shape}")
print(f"  Class 1: {(y_train_smote==1).sum()}, Class 0: {(y_train_smote==0).sum()}")
print(f"  Class distribution: {y_train_smote.value_counts(normalize=True).to_dict()}")

rf_smote = RandomForestClassifier(random_state=42)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)

# Calculate F1 score for minority class (>50K, which is class 1)
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)
print(f"\nF1 Score (minority class >50K): {f1_smote:.6f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote, target_names=['<=50K', '>50K']))

# ============================================
# Comparison
# ============================================
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

f1_difference = f1_smote - f1_baseline
print(f"\nF1 Score (Baseline - No Resampling): {f1_baseline:.6f}")
print(f"F1 Score (SMOTE Resampling):        {f1_smote:.6f}")
print(f"Difference (SMOTE - Baseline):      {f1_difference:.6f}")
print(f"Absolute Difference:                {abs(f1_difference):.6f}")
print(f"\nThreshold: 0.02")
print(f"Is difference > 0.02? {abs(f1_difference) > 0.02}")

if f1_difference > 0.02:
    direction = "SMOTE improves F1 score"
elif f1_difference < -0.02:
    direction = "SMOTE reduces F1 score"
else:
    direction = "No substantial difference (within ±0.02)"

print(f"\nConclusion: {direction}")

# ============================================
# Write results
# ============================================
results = {
    "hypothesis_id": "H5",
    "summary": f"SMOTE oversampling {'increased' if f1_difference > 0 else 'decreased'} the minority-class F1 score by {abs(f1_difference):.6f}, which is {'more' if abs(f1_difference) > 0.02 else 'less'} than the 0.02 threshold. F1 score without SMOTE: {f1_baseline:.6f}, with SMOTE: {f1_smote:.6f}.",
    "primary_metric_name": "F1 score difference (SMOTE - No Resampling) for minority class (>50K)",
    "primary_metric_value": round(f1_difference, 6),
    "direction": direction,
    "methodological_choices": (
        "Random Forest with default hyperparameters (RandomForestClassifier()). "
        "Preprocessing: dropped rows with missing values, label-encoded categorical features. "
        "Train-test split: 70-30 with stratification (random_state=42). "
        "SMOTE applied only to training set. "
        "Evaluation metric: F1 score for minority class (>50K, class=1) on held-out test set. "
        "SMOTE configuration: default parameters (random_state=42). "
        "Comparison: absolute difference in F1 scores between models with and without SMOTE."
    )
}

with open('result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("Results written to result.json")
print("="*60)
print(json.dumps(results, indent=2))
