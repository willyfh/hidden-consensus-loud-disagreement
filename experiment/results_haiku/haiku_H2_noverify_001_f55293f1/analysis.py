import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import json
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names and types:")
print(df.dtypes)
print("\nTarget distribution:")
print(df['class'].value_counts())
print("\nMissing values:")
print(df.isnull().sum())

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target: convert to binary (0 for <=50K, 1 for >50K)
y_encoded = (y == '>50K').astype(int)

print(f"\nTarget encoded distribution: {np.bincount(y_encoded)}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Preprocess data
# For categorical: LabelEncode (not OneHotEncode, to preserve tree structure interpretability and reduce dimensionality)
# For numerical: keep as is (LogisticRegression will handle scaling internally via its solver)

# Create label encoders for categorical variables
X_processed = X.copy()
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

print("\nData after encoding:")
print(X_processed.head())

# Set up stratified 5-fold cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Initialize models with scikit-learn defaults
rf_model = RandomForestClassifier()  # defaults: n_estimators=100, max_depth=None, random_state=None
lr_model = LogisticRegression(max_iter=1000, random_state=42)  # max_iter increased to ensure convergence

print("\nRandom Forest parameters:")
print(rf_model.get_params())
print("\nLogistic Regression parameters:")
print(lr_model.get_params())

# Evaluate models using stratified 5-fold cross-validation with ROC-AUC
print("\n" + "="*60)
print("Cross-validated ROC-AUC Scores (Stratified 5-Fold)")
print("="*60)

rf_scores = cross_val_score(rf_model, X_processed, y_encoded, cv=skf, scoring='roc_auc')
print(f"\nRandom Forest ROC-AUC scores per fold: {rf_scores}")
print(f"Random Forest Mean ROC-AUC: {rf_scores.mean():.6f} (+/- {rf_scores.std():.6f})")

lr_scores = cross_val_score(lr_model, X_processed, y_encoded, cv=skf, scoring='roc_auc')
print(f"\nLogistic Regression ROC-AUC scores per fold: {lr_scores}")
print(f"Logistic Regression Mean ROC-AUC: {lr_scores.mean():.6f} (+/- {lr_scores.std():.6f})")

# Calculate the difference
auc_difference = rf_scores.mean() - lr_scores.mean()
print(f"\nDifference (RF - LogReg): {auc_difference:.6f}")

# Determine the winner
if auc_difference > 0:
    winner = "Random Forest"
    direction = "RF > LogReg"
else:
    winner = "Logistic Regression"
    direction = "RF < LogReg"

print(f"\nWinner: {winner}")
print("="*60)

# Prepare result data
result = {
    "hypothesis_id": "H2",
    "summary": f"Random Forest achieves a mean ROC-AUC of {rf_scores.mean():.6f} while Logistic Regression achieves {lr_scores.mean():.6f} in stratified 5-fold cross-validation. Random Forest {'performs better' if auc_difference > 0 else 'performs worse'} than Logistic Regression by {abs(auc_difference):.6f}.",
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": float(auc_difference),
    "direction": direction,
    "methodological_choices": (
        "Model selection: scikit-learn RandomForestClassifier and LogisticRegression with default parameters (RF: n_estimators=100, max_depth=None; LR: default solver/C). "
        "Encoding: LabelEncoded categorical variables (16 categorical, 6 numerical features). "
        "Validation: Stratified 5-fold cross-validation with random_state=42 to ensure consistent fold splitting. "
        "Evaluation metric: ROC-AUC (area under the receiver operating characteristic curve). "
        "Target encoding: binary (1 for >50K, 0 for <=50K). "
        "No missing value imputation applied (dataset contains no missing values). "
        "LogisticRegression configured with max_iter=1000 to ensure convergence."
    )
}

# Save results to JSON
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResults saved to result.json")

# Additional analysis: feature importance from Random Forest
print("\n" + "="*60)
print("Feature Importance Analysis (from Random Forest)")
print("="*60)

# Train a model on full data to get feature importances
rf_full = RandomForestClassifier(random_state=42)
rf_full.fit(X_processed, y_encoded)

feature_importance = pd.DataFrame({
    'feature': X_processed.columns,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
print(feature_importance.head(10))
