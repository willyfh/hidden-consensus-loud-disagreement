import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
import json
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('adult_income.csv')

print(f"Dataset shape: {df.shape}")
print(f"Target class distribution:\n{df['class'].value_counts()}")

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Convert target to binary (1 for >50K, 0 for <=50K)
y = (y == '>50K').astype(int)

print(f"\nMissing values per column:\n{X.isnull().sum()[X.isnull().sum() > 0]}")

# Preprocess: handle missing values and encode categorical features
# Replace empty strings and missing values with NaN, then fill with mode for categorical
X = X.replace('', np.nan)

# Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\nNumeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# Fill missing values in categorical columns with the mode
for col in categorical_cols:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0]
        X[col].fillna(mode_val, inplace=True)

# Fill missing values in numeric columns with the median
for col in numeric_cols:
    if X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)

# Encode categorical features using LabelEncoder
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print(f"\nFeature shape after preprocessing: {X.shape}")
print(f"Target class balance: {y.value_counts()}")

# Set up stratified 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize models with scikit-learn defaults
rf_model = RandomForestClassifier()
lr_model = LogisticRegression(max_iter=1000)

# Compute cross-validated ROC-AUC scores
print("\n" + "="*60)
print("Computing stratified 5-fold cross-validated ROC-AUC scores")
print("="*60)

rf_scores = cross_val_score(rf_model, X, y, cv=skf, scoring='roc_auc')
lr_scores = cross_val_score(lr_model, X, y, cv=skf, scoring='roc_auc')

rf_mean_auc = rf_scores.mean()
rf_std_auc = rf_scores.std()

lr_mean_auc = lr_scores.mean()
lr_std_auc = lr_scores.std()

auc_difference = rf_mean_auc - lr_mean_auc

print(f"\nRandom Forest ROC-AUC:")
print(f"  Mean: {rf_mean_auc:.6f} (+/- {rf_std_auc:.6f})")
print(f"  Fold scores: {rf_scores}")

print(f"\nLogistic Regression ROC-AUC:")
print(f"  Mean: {lr_mean_auc:.6f} (+/- {lr_std_auc:.6f})")
print(f"  Fold scores: {lr_scores}")

print(f"\nROC-AUC Difference (RF - LogReg): {auc_difference:.6f}")

# Determine the direction
if rf_mean_auc > lr_mean_auc:
    direction = "RF > LogReg"
    summary = f"Random Forest achieves higher stratified 5-fold cross-validated ROC-AUC ({rf_mean_auc:.4f}) compared to Logistic Regression ({lr_mean_auc:.4f}), with a difference of {auc_difference:.4f}."
elif rf_mean_auc < lr_mean_auc:
    direction = "RF < LogReg"
    summary = f"Logistic Regression achieves higher stratified 5-fold cross-validated ROC-AUC ({lr_mean_auc:.4f}) compared to Random Forest ({rf_mean_auc:.4f}), with a difference of {-auc_difference:.4f}."
else:
    direction = "RF = LogReg"
    summary = f"Random Forest and Logistic Regression achieve approximately equal stratified 5-fold cross-validated ROC-AUC ({rf_mean_auc:.4f})."

# Prepare result dictionary
result = {
    "hypothesis_id": "H2",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(auc_difference, 6),
    "direction": direction,
    "methodological_choices": (
        "Preprocessing: Categorical features encoded with LabelEncoder; missing values in categorical columns filled with mode, numeric columns with median. "
        "Model training: RandomForestClassifier() and LogisticRegression(max_iter=1000) using scikit-learn defaults. "
        "Validation: StratifiedKFold with 5 splits (shuffle=True, random_state=42). "
        "Metric: ROC-AUC computed via cross_val_score with scoring='roc_auc'. "
        "Target encoding: Binary (1 for >50K, 0 for <=50K). "
        "No explicit imbalance handling applied beyond stratified sampling."
    )
}

# Save results to result.json
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*60)
print("Results saved to result.json")
print("="*60)
print(json.dumps(result, indent=2))
