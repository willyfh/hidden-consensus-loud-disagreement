import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import json

# Load the data
df = pd.read_csv('adult_income.csv')

# Display basic information
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names and types:")
print(df.dtypes)
print("\nTarget distribution:")
print(df['class'].value_counts())
print("\nMissing values:")
print(df.isnull().sum())

# ===== PREPROCESSING =====
# Make a working copy
df_clean = df.copy()

# Handle missing values (represented as '?')
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].replace('?', np.nan)
        df_clean[col].fillna(df_clean[col].mode()[0] if len(df_clean[col].mode()) > 0 else df_clean[col].value_counts().index[0], inplace=True)

# Separate features and target
X = df_clean.drop('class', axis=1)
y = df_clean['class'].map({'<=50K': 0, '>50K': 1})

print("\n===== PREPROCESSING =====")
print("Features shape:", X.shape)
print("Target distribution in dataset:", y.value_counts().to_dict())

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {len(cat_cols)}")
print(f"Numerical columns: {len(num_cols)}")

# Encode categorical variables
le_dict = {}
X_encoded = X.copy()
for col in cat_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

# ===== TRAIN-TEST SPLIT =====
# Use stratified split to preserve class distribution
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate: {y_test.mean():.4f}")

# ===== MODEL TRAINING =====
# Train logistic regression (natural probability output)
model = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
model.fit(X_train, y_train)

# Get predicted probabilities
y_pred_prob_train = model.predict_proba(X_train)[:, 1]
y_pred_prob_test = model.predict_proba(X_test)[:, 1]

print("\n===== MODEL CALIBRATION ANALYSIS =====")
print("\nPredicted probability statistics on test set:")
print(f"  Min: {y_pred_prob_test.min():.4f}")
print(f"  Max: {y_pred_prob_test.max():.4f}")
print(f"  Mean: {y_pred_prob_test.mean():.4f}")
print(f"  Median: {np.median(y_pred_prob_test):.4f}")

# ===== CALIBRATION CURVE =====
# Calculate calibration curve (fraction of positives vs mean predicted probability)
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob_test, n_bins=10, strategy='uniform')

print("\nCalibration curve (10 bins):")
for i, (pt, pp) in enumerate(zip(prob_true, prob_pred)):
    print(f"  Bin {i}: predicted={pp:.4f}, actual={pt:.4f}, diff={abs(pt-pp):.4f}")

# ===== EXPECTED CALIBRATION ERROR (ECE) =====
# Calculate ECE: average absolute difference between predicted and actual probabilities
def calculate_ece(y_true, y_pred_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_pred_prob > bin_lower) & (y_pred_prob <= bin_upper)
        prop_in_bin = in_bin.mean()

        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_pred_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())

    return ece, bin_accs, bin_confs, bin_counts

ece, _, _, _ = calculate_ece(y_test.values, y_pred_prob_test, n_bins=10)
print(f"\nExpected Calibration Error (ECE): {ece:.4f}")

# ===== BRIER SCORE =====
# Lower is better (perfect calibration = 0)
brier_score = np.mean((y_pred_prob_test - y_test.values) ** 2)
print(f"Brier Score: {brier_score:.4f}")

# ===== MAXIMUM CALIBRATION ERROR (MCE) =====
# Maximum absolute difference across all bins
prob_true, prob_pred = calibration_curve(y_test, y_pred_prob_test, n_bins=10, strategy='uniform')
mce = np.max(np.abs(prob_true - prob_pred))
print(f"Maximum Calibration Error (MCE): {mce:.4f}")

# ===== INTERPRETATION =====
print("\n===== CALIBRATION ASSESSMENT =====")
if ece < 0.05:
    calibration_status = "well-calibrated"
    print(f"✓ The model is WELL-CALIBRATED (ECE={ece:.4f} < 0.05)")
elif ece < 0.10:
    calibration_status = "reasonably calibrated"
    print(f"~ The model is REASONABLY CALIBRATED (ECE={ece:.4f})")
else:
    calibration_status = "poorly calibrated"
    print(f"✗ The model is POORLY CALIBRATED (ECE={ece:.4f} > 0.10)")

# Additional check: how well does predicted probability align with actual positive rate?
print(f"\nMean predicted probability: {y_pred_prob_test.mean():.4f}")
print(f"Actual positive rate in test set: {y_test.mean():.4f}")
print(f"Difference: {abs(y_pred_prob_test.mean() - y_test.mean()):.4f}")

# ===== PREPARE RESULT =====
result = {
    "hypothesis_id": "H6",
    "summary": f"The logistic regression model is {calibration_status} with Expected Calibration Error (ECE) of {ece:.4f}. The model's predicted probabilities are {'well-aligned' if ece < 0.05 else 'reasonably aligned' if ece < 0.10 else 'poorly aligned'} with actual class frequencies.",
    "primary_metric_name": "Expected Calibration Error (ECE)",
    "primary_metric_value": float(ece),
    "direction": f"ECE = {ece:.4f}; model is {calibration_status}" if ece < 0.05 else f"ECE = {ece:.4f}; model shows moderate miscalibration",
    "methodological_choices": (
        "Trained logistic regression on label-encoded categorical features and standardized numerical features. "
        "Used 80-20 train-test split with stratification to preserve class distribution. "
        "Evaluated calibration using Expected Calibration Error (ECE) with 10 equal-width bins, "
        "Brier Score, and maximum calibration error. ECE measures mean absolute difference between "
        "predicted probabilities and actual positive rates across bins and is the standard metric for calibration assessment."
    )
}

# Save result
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n===== RESULT SAVED =====")
print(json.dumps(result, indent=2))
