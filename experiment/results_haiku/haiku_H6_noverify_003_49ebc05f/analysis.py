import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading data...")
df = pd.read_csv('adult_income.csv')

print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nTarget distribution:")
print(df['class'].value_counts())

# Data preprocessing
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

df_clean = df.copy()

# Handle missing values - fill NaN with mode for categorical columns
for col in df_clean.columns:
    if df_clean[col].isnull().any():
        if df_clean[col].dtype == 'object':
            mode_val = df_clean[col].mode()[0]
            df_clean[col] = df_clean[col].fillna(mode_val)
            print(f"Filled NaN in '{col}' with mode: {mode_val}")

# Strip whitespace from string columns
for col in df_clean.columns:
    if df_clean[col].dtype == 'object':
        df_clean[col] = df_clean[col].str.strip()

# Separate features and target
X = df_clean.drop('class', axis=1)
y = (df_clean['class'] == '>50K').astype(int)

print(f"\nFeatures shape: {X.shape}")
print(f"Target distribution (0: <=50K, 1: >50K):")
print(y.value_counts())
print(f"Class balance: {y.mean():.4f} positive")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Encode categorical variables
X_encoded = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col])
    label_encoders[col] = le

print(f"\nEncoded features shape: {X_encoded.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate: {y_test.mean():.4f}")

# Scale numerical features
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test_scaled[numerical_cols] = scaler.transform(X_test[numerical_cols])

# Train an uncalibrated model (Logistic Regression)
print("\n" + "="*60)
print("MODEL TRAINING")
print("="*60)

print("\nTraining Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)

# Get probability predictions
y_prob_train = lr_model.predict_proba(X_train_scaled)[:, 1]
y_prob_test = lr_model.predict_proba(X_test_scaled)[:, 1]

print(f"Train accuracy: {lr_model.score(X_train_scaled, y_train):.4f}")
print(f"Test accuracy: {lr_model.score(X_test_scaled, y_test):.4f}")

# Calibration assessment
print("\n" + "="*60)
print("CALIBRATION ANALYSIS")
print("="*60)

# Calculate Expected Calibration Error (ECE)
def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_sums = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)

    for i in range(len(y_true)):
        bin_idx = int(y_prob[i] * n_bins)
        if bin_idx == n_bins:
            bin_idx = n_bins - 1

        bin_sums[bin_idx] += y_prob[i]
        bin_true[bin_idx] += y_true[i]
        bin_total[bin_idx] += 1

    ece = 0
    for i in range(n_bins):
        if bin_total[i] > 0:
            confidence = bin_sums[i] / bin_total[i]
            accuracy = bin_true[i] / bin_total[i]
            ece += abs(confidence - accuracy) * bin_total[i] / len(y_true)

    return ece

# Calculate Maximum Calibration Error (MCE)
def calculate_mce(y_true, y_prob, n_bins=10):
    """Calculate Maximum Calibration Error"""
    bin_sums = np.zeros(n_bins)
    bin_true = np.zeros(n_bins)
    bin_total = np.zeros(n_bins)

    for i in range(len(y_true)):
        bin_idx = int(y_prob[i] * n_bins)
        if bin_idx == n_bins:
            bin_idx = n_bins - 1

        bin_sums[bin_idx] += y_prob[i]
        bin_true[bin_idx] += y_true[i]
        bin_total[bin_idx] += 1

    mce = 0
    for i in range(n_bins):
        if bin_total[i] > 0:
            confidence = bin_sums[i] / bin_total[i]
            accuracy = bin_true[i] / bin_total[i]
            mce = max(mce, abs(confidence - accuracy))

    return mce

# Calculate calibration metrics on test set
ece = calculate_ece(y_test.values, y_prob_test, n_bins=10)
mce = calculate_mce(y_test.values, y_prob_test, n_bins=10)

# Brier score
brier_score = np.mean((y_prob_test - y_test.values) ** 2)

# Log loss
log_loss = -np.mean(y_test.values * np.log(y_prob_test + 1e-15) + (1 - y_test.values) * np.log(1 - y_prob_test + 1e-15))

print(f"\nUncalibrated Model (Logistic Regression) - Test Set:")
print(f"Expected Calibration Error (ECE): {ece:.4f}")
print(f"Maximum Calibration Error (MCE): {mce:.4f}")
print(f"Brier Score: {brier_score:.4f}")
print(f"Log Loss: {log_loss:.4f}")

# Calibration curve analysis
prob_true, prob_pred = calibration_curve(y_test, y_prob_test, n_bins=10)

print(f"\nCalibration curve analysis (bin-by-bin):")
print("Predicted Prob | Actual Freq")
for pred, true in zip(prob_pred, prob_true):
    diff = abs(pred - true)
    print(f"{pred:.4f}        | {true:.4f}  (diff: {diff:.4f})")

# Try calibration methods
print("\n" + "="*60)
print("CALIBRATION METHODS")
print("="*60)

# Isotonic calibration
print("\nApplying Isotonic Calibration...")
iso_model = CalibratedClassifierCV(lr_model, method='isotonic', cv=5)
iso_model.fit(X_train_scaled, y_train)
y_prob_test_iso = iso_model.predict_proba(X_test_scaled)[:, 1]

ece_iso = calculate_ece(y_test.values, y_prob_test_iso, n_bins=10)
mce_iso = calculate_mce(y_test.values, y_prob_test_iso, n_bins=10)
brier_score_iso = np.mean((y_prob_test_iso - y_test.values) ** 2)
log_loss_iso = -np.mean(y_test.values * np.log(y_prob_test_iso + 1e-15) + (1 - y_test.values) * np.log(1 - y_prob_test_iso + 1e-15))

print(f"ECE (Isotonic): {ece_iso:.4f}")
print(f"MCE (Isotonic): {mce_iso:.4f}")
print(f"Brier Score (Isotonic): {brier_score_iso:.4f}")
print(f"Log Loss (Isotonic): {log_loss_iso:.4f}")

# Sigmoid (Platt) calibration
print("\nApplying Sigmoid (Platt) Calibration...")
sig_model = CalibratedClassifierCV(lr_model, method='sigmoid', cv=5)
sig_model.fit(X_train_scaled, y_train)
y_prob_test_sig = sig_model.predict_proba(X_test_scaled)[:, 1]

ece_sig = calculate_ece(y_test.values, y_prob_test_sig, n_bins=10)
mce_sig = calculate_mce(y_test.values, y_prob_test_sig, n_bins=10)
brier_score_sig = np.mean((y_prob_test_sig - y_test.values) ** 2)
log_loss_sig = -np.mean(y_test.values * np.log(y_prob_test_sig + 1e-15) + (1 - y_test.values) * np.log(1 - y_prob_test_sig + 1e-15))

print(f"ECE (Sigmoid): {ece_sig:.4f}")
print(f"MCE (Sigmoid): {mce_sig:.4f}")
print(f"Brier Score (Sigmoid): {brier_score_sig:.4f}")
print(f"Log Loss (Sigmoid): {log_loss_sig:.4f}")

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

print("\nModel Calibration Assessment:")
print(f"Model: Logistic Regression (uncalibrated)")
print(f"\nPrimary Metric: Expected Calibration Error (ECE)")
print(f"ECE Value: {ece:.4f}")

# Interpretation
if ece < 0.05:
    calibration_status = "Excellent calibration"
elif ece < 0.10:
    calibration_status = "Good calibration"
elif ece < 0.15:
    calibration_status = "Moderate calibration"
else:
    calibration_status = "Poor calibration"

print(f"\nInterpretation: {calibration_status}")
print(f"\nThe model is {'WELL-CALIBRATED' if ece < 0.10 else 'NOT WELL-CALIBRATED'}")
print(f"\nAdditional metrics:")
print(f"- Maximum Calibration Error: {mce:.4f}")
print(f"- Brier Score: {brier_score:.4f}")
print(f"- Log Loss: {log_loss:.4f}")

print(f"\nCalibration can be improved with:")
print(f"- Isotonic calibration reduces ECE to {ece_iso:.4f}")
print(f"- Sigmoid calibration reduces ECE to {ece_sig:.4f}")

# Determine primary finding
is_well_calibrated = ece < 0.10
direction = "Model is well-calibrated" if is_well_calibrated else "Model is poorly calibrated"

print("\n" + "="*60)
print("RESULT")
print("="*60)
print(f"\nHypothesis: Is the model well-calibrated?")
print(f"Finding: {direction}")
print(f"ECE: {ece:.4f}")
