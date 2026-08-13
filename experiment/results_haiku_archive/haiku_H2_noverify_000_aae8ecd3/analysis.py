import pandas as pd
import numpy as np
import json
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
df = pd.read_csv('adult_income.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nTarget distribution:\n{df['class'].value_counts()}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Explore data structure
print(f"\nFirst few rows:")
print(df.head())

# Prepare data for modeling
print("\nPreparing data for modeling...")

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"Target classes: {le_target.classes_}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical variables
X_processed = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Handle missing values in categorical columns
    if X_processed[col].isnull().any():
        X_processed[col] = X_processed[col].fillna('MISSING')
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

# Handle missing values in numerical columns
X_processed[numerical_cols] = X_processed[numerical_cols].fillna(X_processed[numerical_cols].mean())

print(f"\nProcessed data shape: {X_processed.shape}")
print(f"Final feature matrix:\n{X_processed.head()}")

# Define models
print("\n" + "="*60)
print("Training models with 5-fold stratified cross-validation...")
print("="*60)

# Logistic regression pipeline (with scaling)
lr_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('lr', LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1))
])

# Random forest (no scaling needed)
rf_model = RandomForestClassifier(random_state=42, n_jobs=-1)

# Define stratified k-fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Calculate ROC-AUC scores using cross-validation
print("\nCalculating ROC-AUC for Logistic Regression...")
lr_roc_auc = cross_val_score(lr_pipeline, X_processed, y_encoded,
                              cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"Logistic Regression ROC-AUC scores: {lr_roc_auc}")
print(f"Mean ROC-AUC: {lr_roc_auc.mean():.6f} (+/- {lr_roc_auc.std():.6f})")

print("\nCalculating ROC-AUC for Random Forest...")
rf_roc_auc = cross_val_score(rf_model, X_processed, y_encoded,
                              cv=skf, scoring='roc_auc', n_jobs=-1)
print(f"Random Forest ROC-AUC scores: {rf_roc_auc}")
print(f"Mean ROC-AUC: {rf_roc_auc.mean():.6f} (+/- {rf_roc_auc.std():.6f})")

# Calculate difference
lr_mean = lr_roc_auc.mean()
rf_mean = rf_roc_auc.mean()
difference = rf_mean - lr_mean

print("\n" + "="*60)
print("COMPARISON RESULTS")
print("="*60)
print(f"Logistic Regression Mean ROC-AUC: {lr_mean:.6f}")
print(f"Random Forest Mean ROC-AUC: {rf_mean:.6f}")
print(f"Difference (RF - LogReg): {difference:.6f}")
print(f"RF ROC-AUC is {'HIGHER' if difference > 0 else 'LOWER' if difference < 0 else 'EQUAL'} than LogReg")

# Determine direction
if difference > 0:
    direction = "RF > LogReg"
    conclusion = "Yes, Random Forest achieves higher ROC-AUC than Logistic Regression."
else:
    direction = "RF <= LogReg"
    conclusion = "No, Random Forest does not achieve higher ROC-AUC than Logistic Regression."

# Prepare results
result = {
    "hypothesis_id": "H2",
    "summary": conclusion,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(difference, 6),
    "direction": direction,
    "methodological_choices": (
        "Label-encoded all categorical features using sklearn's LabelEncoder. "
        "Filled missing numerical values with column mean. "
        "Used sklearn defaults for both models: LogisticRegression(max_iter=1000) with StandardScaler pipeline "
        "and RandomForestClassifier() without scaling. "
        "Evaluated via 5-fold stratified cross-validation with random_state=42. "
        "Metric: ROC-AUC (sklearn's roc_auc_score). "
        "No class imbalance handling applied; models evaluated on full dataset with target distribution: "
        f"{dict(y.value_counts())}. "
        f"Final dataset: {X_processed.shape[0]} samples × {X_processed.shape[1]} features "
        f"({len(categorical_cols)} categorical, {len(numerical_cols)} numerical)."
    )
}

# Save results
print("\nSaving results to result.json...")
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("Results saved successfully!")
print("\nFinal Result:")
print(json.dumps(result, indent=2))
