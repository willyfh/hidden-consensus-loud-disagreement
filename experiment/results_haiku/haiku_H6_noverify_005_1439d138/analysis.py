import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import json
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv('adult_income.csv')

# Prepare the data
# Create a copy for processing
data = df.copy()

# Encode target variable
data['class'] = (data['class'] == '>50K').astype(int)

# Handle missing values
data['workclass'].fillna(data['workclass'].mode()[0], inplace=True)
data['occupation'].fillna(data['occupation'].mode()[0], inplace=True)
data['native-country'].fillna(data['native-country'].mode()[0], inplace=True)

# Separate features and target
X = data.drop('class', axis=1)
y = data['class']

# Encode categorical variables
categorical_cols = X.select_dtypes(include=['object']).columns
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

# Train-test split (80-20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression model
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_scaled, y_train)

# Get predicted probabilities on test set
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = model.predict(X_test_scaled)

# Calculate calibration metrics
brier_score = brier_score_loss(y_test, y_pred_proba)
log_loss_val = log_loss(y_test, y_pred_proba)

print("=" * 60)
print("CALIBRATION ANALYSIS RESULTS")
print("=" * 60)
print(f"\nBrier Score (lower is better): {brier_score:.4f}")
print(f"Log Loss (lower is better): {log_loss_val:.4f}")

# Calculate Expected Calibration Error (ECE)
# Divide predictions into bins and compare mean prediction with actual frequency
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
bin_indices = np.digitize(y_pred_proba, bin_edges) - 1
bin_indices = np.clip(bin_indices, 0, n_bins - 1)

ece = 0
mce = 0
calibration_data = []

for bin_idx in range(n_bins):
    mask = bin_indices == bin_idx
    if mask.sum() > 0:
        bin_accuracy = y_test[mask].mean()
        bin_confidence = y_pred_proba[mask].mean()
        bin_count = mask.sum()

        calibration_data.append({
            'bin': bin_idx,
            'bin_lower': bin_edges[bin_idx],
            'bin_upper': bin_edges[bin_idx + 1],
            'bin_accuracy': bin_accuracy,
            'bin_confidence': bin_confidence,
            'bin_count': bin_count,
            'abs_diff': abs(bin_accuracy - bin_confidence)
        })

        ece += mask.sum() / len(y_test) * abs(bin_accuracy - bin_confidence)
        mce = max(mce, abs(bin_accuracy - bin_confidence))

print(f"\nExpected Calibration Error (ECE): {ece:.4f}")
print(f"Maximum Calibration Error (MCE): {mce:.4f}")

print("\nCalibration by probability bins:")
print("Bin Range\t\tConfidence\tAccuracy\tCount\t|Diff|")
print("-" * 70)
for cal in calibration_data:
    print(f"[{cal['bin_lower']:.1f}-{cal['bin_upper']:.1f}]\t\t"
          f"{cal['bin_confidence']:.4f}\t\t{cal['bin_accuracy']:.4f}\t\t"
          f"{cal['bin_count']}\t{cal['abs_diff']:.4f}")

# Test set statistics
print(f"\nTest set statistics:")
print(f"  Total samples: {len(y_test)}")
print(f"  Positive class: {y_test.sum()} ({y_test.mean():.4f})")
print(f"  Negative class: {(1-y_test).sum()} ({(1-y_test).mean():.4f})")
print(f"  Mean predicted probability: {y_pred_proba.mean():.4f}")
print(f"  Std of predicted probability: {y_pred_proba.std():.4f}")

# Determine calibration assessment
print("\n" + "=" * 60)
print("CALIBRATION ASSESSMENT")
print("=" * 60)

# Generally:
# ECE < 0.10 = well calibrated
# ECE 0.10-0.20 = reasonably calibrated
# ECE > 0.20 = poorly calibrated

if ece < 0.10:
    calibration_status = "well-calibrated"
    interpretation = "The model predictions closely match actual probabilities"
elif ece < 0.20:
    calibration_status = "reasonably calibrated"
    interpretation = "The model predictions moderately match actual probabilities, with some systematic bias"
else:
    calibration_status = "poorly calibrated"
    interpretation = "The model predictions do not reliably match actual probabilities"

print(f"\nStatus: {calibration_status}")
print(f"ECE: {ece:.4f}")
print(f"Interpretation: {interpretation}")

# Prepare findings for JSON
result = {
    "hypothesis_id": "H6",
    "summary": f"The logistic regression model is {calibration_status} with an Expected Calibration Error (ECE) of {ece:.4f}. " +
               ("The predicted probabilities closely align with actual class frequencies, indicating reliable probability predictions."
                if ece < 0.10 else
                "The predicted probabilities show some systematic deviation from actual class frequencies."),
    "primary_metric_name": "Expected Calibration Error (ECE)",
    "primary_metric_value": round(ece, 4),
    "direction": f"Model is {calibration_status} (ECE: {ece:.4f})",
    "methodological_choices": (
        "Logistic Regression model with L2 regularization (default C=1.0). "
        "Data preprocessing: missing values imputed with mode for categorical features (workclass, occupation, native-country). "
        "Categorical features encoded using LabelEncoder. Numeric features scaled with StandardScaler. "
        "Train-test split 80-20 with stratification. Calibration assessed using 10-bin Expected Calibration Error (ECE), "
        "Maximum Calibration Error (MCE), Brier Score, and Log Loss on test set (n=9769)."
    )
}

# Save results
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResults saved to result.json")
