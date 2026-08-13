import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
import json

# Load the dataset
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget class distribution:")
print(df['class'].value_counts())

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print("\nTarget encoding: ", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object'], exclude=[]).columns.tolist()

print("\nNumeric columns:", numeric_cols)
print("Categorical columns:", categorical_cols)

# Preprocessing
X_processed = X.copy()

# Handle missing values
# For numeric: fill with median
for col in numeric_cols:
    X_processed[col] = X_processed[col].fillna(X_processed[col].median())

# For categorical: fill with mode or a placeholder
for col in categorical_cols:
    if X_processed[col].isnull().sum() > 0:
        X_processed[col] = X_processed[col].fillna('Unknown')

print("\nMissing values after imputation:")
print(X_processed.isnull().sum())

# Encode categorical variables
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    le_dict[col] = le

print("\nProcessed data shape:", X_processed.shape)
print("\nFirst few rows of processed data:")
print(X_processed.head())

# Setup stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model 1: Logistic Regression with scikit-learn defaults
print("\n" + "="*60)
print("Training Logistic Regression with scikit-learn defaults")
print("="*60)

# Scale features for logistic regression (it's sensitive to feature scaling)
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])

lr_scores = cross_val_score(pipeline_lr, X_processed, y_encoded,
                            cv=skf, scoring='roc_auc')
print(f"Logistic Regression ROC-AUC scores: {lr_scores}")
print(f"Mean ROC-AUC: {lr_scores.mean():.6f} (+/- {lr_scores.std():.6f})")

# Model 2: Random Forest with scikit-learn defaults
print("\n" + "="*60)
print("Training Random Forest with scikit-learn defaults")
print("="*60)

rf_model = RandomForestClassifier()
rf_scores = cross_val_score(rf_model, X_processed, y_encoded,
                            cv=skf, scoring='roc_auc')
print(f"Random Forest ROC-AUC scores: {rf_scores}")
print(f"Mean ROC-AUC: {rf_scores.mean():.6f} (+/- {rf_scores.std():.6f})")

# Compare results
print("\n" + "="*60)
print("COMPARISON")
print("="*60)
lr_mean = lr_scores.mean()
rf_mean = rf_scores.mean()
difference = rf_mean - lr_mean

print(f"\nLogistic Regression mean ROC-AUC: {lr_mean:.6f}")
print(f"Random Forest mean ROC-AUC: {rf_mean:.6f}")
print(f"Difference (RF - LogReg): {difference:.6f}")

if rf_mean > lr_mean:
    print(f"\n✓ Random Forest achieves HIGHER ROC-AUC by {abs(difference):.6f}")
    direction = "RF > LogReg"
else:
    print(f"\n✗ Logistic Regression achieves HIGHER ROC-AUC by {abs(difference):.6f}")
    direction = "LogReg > RF"

# Prepare results
result = {
    "hypothesis_id": "H2",
    "summary": f"Random Forest achieved a mean ROC-AUC of {rf_mean:.6f} while Logistic Regression achieved {lr_mean:.6f} in stratified 5-fold cross-validation. {'Random Forest achieves higher performance.' if rf_mean > lr_mean else 'Logistic Regression achieves higher performance.'}",
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": float(difference),
    "direction": direction,
    "methodological_choices": (
        "Preprocessing: Numeric features imputed with median values. Categorical features filled with 'Unknown' for missing values, then label-encoded. "
        "Encoding: All categorical variables (workclass, education, marital-status, occupation, relationship, race, sex, native-country) were label-encoded. "
        "Target: 'class' variable binary encoded (<=50K=0, >50K=1). "
        "Validation: Stratified 5-fold cross-validation with random_state=42. "
        "Logistic Regression: Used Pipeline with StandardScaler followed by LogisticRegression() with scikit-learn defaults (max_iter=100, default solver). "
        "Random Forest: RandomForestClassifier() with scikit-learn defaults (n_estimators=100, no scaling needed). "
        "Metric: ROC-AUC score using cross_val_score with scoring='roc_auc'."
    )
}

# Save results to JSON
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*60)
print("Results saved to result.json")
print("="*60)
print(json.dumps(result, indent=2))
