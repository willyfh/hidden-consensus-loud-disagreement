import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import json
import warnings
warnings.filterwarnings('ignore')

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
label_encoder_y = LabelEncoder()
y_encoded = label_encoder_y.fit_transform(y)

print("\nTarget encoding:", dict(zip(label_encoder_y.classes_, label_encoder_y.transform(label_encoder_y.classes_))))

# Handle categorical and numerical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Encode categorical variables
X_encoded = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Handle potential missing values
    X_encoded[col] = X_encoded[col].fillna('missing')
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le

# Fill missing values in numerical columns if any
X_encoded = X_encoded.fillna(X_encoded.mean())

print("\nFinal feature matrix shape:", X_encoded.shape)
print("Final feature matrix sample:")
print(X_encoded.head())

# Set up stratified k-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest Classifier with scikit-learn defaults
rf_model = RandomForestClassifier(random_state=42)

# Logistic Regression with scikit-learn defaults
lr_model = LogisticRegression(random_state=42, max_iter=1000)

# Calculate stratified 5-fold cross-validated ROC-AUC
print("\n" + "="*60)
print("Computing stratified 5-fold cross-validated ROC-AUC...")
print("="*60)

rf_scores = cross_val_score(rf_model, X_encoded, y_encoded, cv=skf, scoring='roc_auc')
lr_scores = cross_val_score(lr_model, X_encoded, y_encoded, cv=skf, scoring='roc_auc')

print(f"\nRandom Forest ROC-AUC scores (5 folds): {rf_scores}")
print(f"Random Forest mean ROC-AUC: {rf_scores.mean():.6f} (+/- {rf_scores.std():.6f})")

print(f"\nLogistic Regression ROC-AUC scores (5 folds): {lr_scores}")
print(f"Logistic Regression mean ROC-AUC: {lr_scores.mean():.6f} (+/- {lr_scores.std():.6f})")

# Calculate the difference
roc_auc_diff = rf_scores.mean() - lr_scores.mean()
print(f"\nDifference (RF - LogReg): {roc_auc_diff:.6f}")

# Determine if RF > LogReg
if roc_auc_diff > 0:
    direction = "RF > LogReg"
    summary = f"Random Forest achieves a higher stratified 5-fold cross-validated ROC-AUC ({rf_scores.mean():.4f}) compared to Logistic Regression ({lr_scores.mean():.4f}) on the Adult Income dataset."
else:
    direction = "LogReg ≥ RF"
    summary = f"Logistic Regression achieves a higher or equal stratified 5-fold cross-validated ROC-AUC ({lr_scores.mean():.4f}) compared to Random Forest ({rf_scores.mean():.4f}) on the Adult Income dataset."

# Prepare the result
result = {
    "hypothesis_id": "H2",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(roc_auc_diff, 6),
    "direction": direction,
    "methodological_choices": (
        "Used scikit-learn default hyperparameters for both RandomForestClassifier and LogisticRegression. "
        "Encoded categorical variables using LabelEncoder. "
        "Filled missing values in numerical features with column means. "
        "Applied stratified 5-fold cross-validation with random_state=42 for reproducibility. "
        "ROC-AUC computed on the encoded target (0/1). "
        "No additional preprocessing, scaling, or hyperparameter tuning applied beyond defaults."
    )
}

# Save results
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(json.dumps(result, indent=2))
print("\nResults saved to result.json")
