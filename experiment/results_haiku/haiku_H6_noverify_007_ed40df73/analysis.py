"""
Analysis of model calibration for the Adult Income dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')

# Data preprocessing
print("=" * 60)
print("DATA PREPROCESSING")
print("=" * 60)

# Drop rows with missing critical values (workclass, occupation)
df_clean = df.dropna(subset=['workclass', 'occupation'])
print(f"Rows after dropping missing values: {len(df_clean)}")

# Separate features and target
X = df_clean.drop('class', axis=1)
y = df_clean['class']

# Encode target variable
y_encoded = (y == '>50K').astype(int)
print(f"Target distribution: {y_encoded.value_counts().to_dict()}")
print(f"Positive class rate: {y_encoded.mean():.4f}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Encode categorical variables
X_processed = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col])
    label_encoders[col] = le

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nTrain set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate: {y_test.mean():.4f}")

# Standardize numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train models and evaluate calibration
print("\n" + "=" * 60)
print("MODEL TRAINING AND CALIBRATION EVALUATION")
print("=" * 60)

# Model 1: Logistic Regression
print("\n1. LOGISTIC REGRESSION")
print("-" * 60)

lr = LogisticRegression(max_iter=1000, random_state=42, solver='lbfgs')
lr.fit(X_train_scaled, y_train)

# Get predicted probabilities
y_pred_proba_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Calculate calibration metrics
fraction_of_positives_lr, mean_predicted_value_lr = calibration_curve(
    y_test, y_pred_proba_lr, n_bins=10, strategy='uniform'
)

ece_lr = np.mean(np.abs(fraction_of_positives_lr - mean_predicted_value_lr))
brier_lr = brier_score_loss(y_test, y_pred_proba_lr)
logloss_lr = log_loss(y_test, y_pred_proba_lr)

print(f"Expected Calibration Error (ECE): {ece_lr:.4f}")
print(f"Brier Score: {brier_lr:.4f}")
print(f"Log Loss: {logloss_lr:.4f}")
print(f"Predicted probability range: [{y_pred_proba_lr.min():.4f}, {y_pred_proba_lr.max():.4f}]")

# Model 2: Random Forest (typically less calibrated)
print("\n2. RANDOM FOREST (uncalibrated)")
print("-" * 60)

rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)

y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

fraction_of_positives_rf, mean_predicted_value_rf = calibration_curve(
    y_test, y_pred_proba_rf, n_bins=10, strategy='uniform'
)

ece_rf = np.mean(np.abs(fraction_of_positives_rf - mean_predicted_value_rf))
brier_rf = brier_score_loss(y_test, y_pred_proba_rf)
logloss_rf = log_loss(y_test, y_pred_proba_rf)

print(f"Expected Calibration Error (ECE): {ece_rf:.4f}")
print(f"Brier Score: {brier_rf:.4f}")
print(f"Log Loss: {logloss_rf:.4f}")
print(f"Predicted probability range: [{y_pred_proba_rf.min():.4f}, {y_pred_proba_rf.max():.4f}]")

# Model 3: Calibrated Random Forest
print("\n3. RANDOM FOREST (calibrated with Platt scaling)")
print("-" * 60)

rf_calibrated = CalibratedClassifierCV(
    estimator=RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
    method='sigmoid',
    cv=5
)
rf_calibrated.fit(X_train, y_train)

y_pred_proba_rf_cal = rf_calibrated.predict_proba(X_test)[:, 1]

fraction_of_positives_rf_cal, mean_predicted_value_rf_cal = calibration_curve(
    y_test, y_pred_proba_rf_cal, n_bins=10, strategy='uniform'
)

ece_rf_cal = np.mean(np.abs(fraction_of_positives_rf_cal - mean_predicted_value_rf_cal))
brier_rf_cal = brier_score_loss(y_test, y_pred_proba_rf_cal)
logloss_rf_cal = log_loss(y_test, y_pred_proba_rf_cal)

print(f"Expected Calibration Error (ECE): {ece_rf_cal:.4f}")
print(f"Brier Score: {brier_rf_cal:.4f}")
print(f"Log Loss: {logloss_rf_cal:.4f}")
print(f"Predicted probability range: [{y_pred_proba_rf_cal.min():.4f}, {y_pred_proba_rf_cal.max():.4f}]")

# Detailed calibration analysis for each model
print("\n" + "=" * 60)
print("CALIBRATION CURVE ANALYSIS")
print("=" * 60)

print("\nLogistic Regression calibration by bin:")
print("(Predicted Prob -> Actual Positive Rate)")
for i in range(len(mean_predicted_value_lr)):
    print(f"  Bin {i}: {mean_predicted_value_lr[i]:.3f} -> {fraction_of_positives_lr[i]:.3f}")

# Statistical test of calibration (Hosmer-Lemeshow style check)
def calibration_summary(y_true, y_pred_proba, model_name):
    # Bin predictions
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_pred_proba, bin_edges) - 1

    total_error = 0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() == 0:
            continue
        actual_rate = y_true[mask].mean()
        predicted_rate = y_pred_proba[mask].mean()
        total_error += abs(actual_rate - predicted_rate) * mask.sum()

    return total_error / len(y_true)

mae_lr = calibration_summary(y_test.values, y_pred_proba_lr, "LogReg")
mae_rf = calibration_summary(y_test.values, y_pred_proba_rf, "RF")
mae_rf_cal = calibration_summary(y_test.values, y_pred_proba_rf_cal, "RF_Cal")

print("\n" + "=" * 60)
print("SUMMARY OF CALIBRATION METRICS")
print("=" * 60)

summary_data = {
    'Model': ['Logistic Regression', 'Random Forest (uncalibrated)', 'Random Forest (calibrated)'],
    'ECE': [ece_lr, ece_rf, ece_rf_cal],
    'Brier Score': [brier_lr, brier_rf, brier_rf_cal],
    'Log Loss': [logloss_lr, logloss_rf, logloss_rf_cal]
}

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Conclusion
print("\n" + "=" * 60)
print("INTERPRETATION")
print("=" * 60)

print("\nECE Interpretation (lower is better):")
print("  ECE < 0.05: Well-calibrated")
print("  0.05 <= ECE < 0.10: Reasonably calibrated")
print("  0.10 <= ECE < 0.20: Poorly calibrated")
print("  ECE >= 0.20: Very poorly calibrated")

print(f"\nLogistic Regression ECE: {ece_lr:.4f} -> ", end="")
if ece_lr < 0.05:
    print("WELL-CALIBRATED")
elif ece_lr < 0.10:
    print("REASONABLY CALIBRATED")
elif ece_lr < 0.20:
    print("POORLY CALIBRATED")
else:
    print("VERY POORLY CALIBRATED")

print(f"\nRandom Forest (uncalibrated) ECE: {ece_rf:.4f} -> ", end="")
if ece_rf < 0.05:
    print("WELL-CALIBRATED")
elif ece_rf < 0.10:
    print("REASONABLY CALIBRATED")
elif ece_rf < 0.20:
    print("POORLY CALIBRATED")
else:
    print("VERY POORLY CALIBRATED")

print(f"\nRandom Forest (calibrated) ECE: {ece_rf_cal:.4f} -> ", end="")
if ece_rf_cal < 0.05:
    print("WELL-CALIBRATED")
elif ece_rf_cal < 0.10:
    print("REASONABLY CALIBRATED")
elif ece_rf_cal < 0.20:
    print("POORLY CALIBRATED")
else:
    print("VERY POORLY CALIBRATED")

# Determine primary metric and finding
print("\n" + "=" * 60)
print("FINAL ASSESSMENT")
print("=" * 60)

# The question asks if "the model" is well-calibrated
# Since no specific model is specified, I'll evaluate the default/typical approach: Logistic Regression
# This is a natural baseline for binary classification

primary_metric = ece_lr
primary_metric_name = "Expected Calibration Error (ECE) - Logistic Regression"

if primary_metric < 0.05:
    direction = "model is well-calibrated"
    summary = f"The logistic regression model is well-calibrated with an ECE of {primary_metric:.4f}, indicating that predicted probabilities closely match actual observed frequencies. Predictions are reliable for probability-based decisions."
elif primary_metric < 0.10:
    direction = "model is reasonably calibrated"
    summary = f"The logistic regression model is reasonably calibrated with an ECE of {primary_metric:.4f}. While there are minor discrepancies between predicted and actual probabilities, the model is suitable for most applications."
else:
    direction = "model calibration requires improvement"
    summary = f"The logistic regression model shows poor calibration with an ECE of {primary_metric:.4f}. Predicted probabilities significantly deviate from actual frequencies and may not be reliable for probability-based decisions."

print(f"Primary Metric: {primary_metric_name}")
print(f"Primary Metric Value: {primary_metric:.4f}")
print(f"Direction: {direction}")
print(f"Summary: {summary}")

# Save results to JSON
results = {
    "hypothesis_id": "H6",
    "summary": summary,
    "primary_metric_name": primary_metric_name,
    "primary_metric_value": round(primary_metric, 4),
    "direction": direction,
    "methodological_choices": (
        "Used logistic regression with standardized numerical features and label-encoded categorical variables. "
        "Train/test split: 70/30 with stratification on target. "
        "Calibration evaluation: Expected Calibration Error (ECE) with 10 bins, supplemented by Brier score and log loss. "
        "ECE measures mean absolute difference between predicted probability and actual positive rate per bin. "
        "Threshold: ECE < 0.05 for 'well-calibrated', 0.05-0.10 for 'reasonably calibrated', >= 0.10 for 'poorly calibrated'. "
        "Also evaluated uncalibrated and calibrated random forests for comparison to establish baseline calibration quality."
    )
}

with open('result.json', 'w') as f:
    json.dump(results, f, indent=2)

print("\nResults saved to result.json")
