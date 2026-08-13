import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score
import json
import warnings

warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df['class'].value_counts())

# Prepare data
X = df.drop('class', axis=1)
y = df['class']

# Encode target variable
le_target = LabelEncoder()
y = le_target.fit_transform(y)

print(f"\nTarget encoded: {le_target.classes_}")

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# Handle missing values and preprocessing
X = X.copy()

# For numeric columns: fill with median
for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

# For categorical columns: fill with mode, then encode
for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Ensure X is numeric
X = X.astype(float)

print(f"\nFinal feature matrix shape: {X.shape}")
print(f"Features: {X.columns.tolist()}")

# Set up stratified 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize models with scikit-learn defaults
rf = RandomForestClassifier(random_state=42)
lr = LogisticRegression(max_iter=1000, random_state=42)

# Perform cross-validation
print("\n" + "="*60)
print("Performing Stratified 5-Fold Cross-Validation")
print("="*60)

# Cross-validate Random Forest
rf_scores = cross_validate(
    rf, X, y,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    return_train_score=True
)

# Cross-validate Logistic Regression
lr_scores = cross_validate(
    lr, X, y,
    cv=skf,
    scoring='roc_auc',
    n_jobs=-1,
    return_train_score=True
)

# Extract test fold scores
rf_test_auc = rf_scores['test_score']
lr_test_auc = lr_scores['test_score']

# Compute means and standard deviations
rf_mean_auc = rf_test_auc.mean()
rf_std_auc = rf_test_auc.std()

lr_mean_auc = lr_test_auc.mean()
lr_std_auc = lr_test_auc.std()

# Compute difference
auc_difference = rf_mean_auc - lr_mean_auc

print(f"\nRandom Forest ROC-AUC (5-fold CV):")
print(f"  Fold scores: {rf_test_auc}")
print(f"  Mean: {rf_mean_auc:.6f}")
print(f"  Std:  {rf_std_auc:.6f}")

print(f"\nLogistic Regression ROC-AUC (5-fold CV):")
print(f"  Fold scores: {lr_test_auc}")
print(f"  Mean: {lr_mean_auc:.6f}")
print(f"  Std:  {lr_std_auc:.6f}")

print(f"\nDifference (RF - LogReg): {auc_difference:.6f}")

# Determine direction
if auc_difference > 0:
    direction = "RF > LogReg"
else:
    direction = "LogReg >= RF"

# Prepare summary
summary = f"Random Forest achieves a stratified 5-fold cross-validated ROC-AUC of {rf_mean_auc:.4f}, while Logistic Regression achieves {lr_mean_auc:.4f}. Random Forest {'outperforms' if auc_difference > 0 else 'does not outperform'} Logistic Regression by {abs(auc_difference):.6f}."

# Methodological choices explanation
methodological_choices = (
    "Preprocessing: Numeric features imputed with median, categorical features imputed with mode, "
    "then label-encoded. All categorical variables (workclass, education, marital-status, occupation, "
    "relationship, race, sex, native-country) encoded as integers. "
    "Validation: Stratified 5-fold cross-validation with random_state=42 to ensure class balance in folds. "
    "Models: Both trained with scikit-learn defaults (RandomForestClassifier(), LogisticRegression(max_iter=1000, random_state=42)). "
    "Metric: ROC-AUC score on test fold. "
    "No feature engineering, scaling, hyperparameter tuning, or class imbalance handling beyond stratification."
)

# Create result dictionary
result = {
    "hypothesis_id": "H2",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(auc_difference, 6),
    "direction": direction,
    "methodological_choices": methodological_choices
}

# Save result
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*60)
print("Results saved to result.json")
print("="*60)
print(json.dumps(result, indent=2))
