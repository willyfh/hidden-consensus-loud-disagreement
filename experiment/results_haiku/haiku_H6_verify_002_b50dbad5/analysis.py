import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('adult_income.csv')

print("=" * 60)
print("DATA EXPLORATION")
print("=" * 60)
print(f"Shape: {df.shape}")
print(f"\nColumns: {df.columns.tolist()}")
print(f"\nTarget distribution:\n{df['class'].value_counts()}")
print(f"Target proportions:\n{df['class'].value_counts(normalize=True)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Preprocessing
print("\n" + "=" * 60)
print("PREPROCESSING")
print("=" * 60)

# Handle missing values (replace '?' with NaN, then drop)
df = df.replace('?', np.nan)
missing_before = df.isnull().sum().sum()
df = df.dropna()
missing_after = df.isnull().sum().sum()
print(f"Rows removed due to missing values: {len(df) - (len(df) + missing_before - missing_after)}")
print(f"Final dataset size: {df.shape[0]}")

# Encode target variable
df['class_binary'] = (df['class'] == '>50K').astype(int)
y = df['class_binary'].values

# Identify categorical and numerical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('class')

numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_cols.remove('class_binary')

print(f"Categorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

# Encode categorical features
X = df[categorical_cols + numerical_cols].copy()
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

X = X.values

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target proportions in full data: {np.mean(y):.4f} (positive class)")

# Train-test split with random seed
print("\n" + "=" * 60)
print("MAIN ANALYSIS: CALIBRATION ON INITIAL SPLIT")
print("=" * 60)

random_seed = 42
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=random_seed, stratify=y
)

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train positive rate: {np.mean(y_train):.4f}")
print(f"Test positive rate: {np.mean(y_test):.4f}")

# Train logistic regression
model = LogisticRegression(max_iter=1000, random_state=random_seed)
model.fit(X_train, y_train)

# Get predicted probabilities
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print(f"\nModel trained (Logistic Regression)")
print(f"Accuracy: {(y_pred == y_test).mean():.4f}")

# Calculate calibration metrics
brier = brier_score_loss(y_test, y_pred_proba)
print(f"Brier Score: {brier:.6f}")

# Expected Calibration Error (ECE)
n_bins = 10
bin_edges = np.linspace(0, 1, n_bins + 1)
ece = 0
bin_counts = 0

for i in range(n_bins):
    mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
    if mask.sum() > 0:
        bin_acc = y_test[mask].mean()
        bin_conf = y_pred_proba[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * mask.sum()
        bin_counts += mask.sum()

ece = ece / len(y_test) if len(y_test) > 0 else 0
print(f"Expected Calibration Error (ECE): {ece:.6f}")

# Maximum Calibration Error (MCE)
mce = 0
for i in range(n_bins):
    mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i+1])
    if mask.sum() > 0:
        bin_acc = y_test[mask].mean()
        bin_conf = y_pred_proba[mask].mean()
        mce = max(mce, np.abs(bin_acc - bin_conf))

print(f"Maximum Calibration Error (MCE): {mce:.6f}")

# Calibration curve
prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins)
print(f"\nCalibration curve (bin midpoints):")
print(f"Mean predicted prob per bin: {prob_pred}")
print(f"Actual frequency per bin: {prob_true}")

print("\n" + "=" * 60)
print("STABILITY CHECK: REPEATED STRATIFIED CROSS-VALIDATION")
print("=" * 60)

# Repeated 5-fold cross-validation with different seeds
n_splits = 5
n_repeats = 5
ece_scores = []
brier_scores = []
mce_scores = []

for repeat in range(n_repeats):
    seed = 42 + repeat

    X_train_cv, X_test_cv, y_train_cv, y_test_cv = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    model_cv = LogisticRegression(max_iter=1000, random_state=seed)
    model_cv.fit(X_train_cv, y_train_cv)
    y_pred_proba_cv = model_cv.predict_proba(X_test_cv)[:, 1]

    # Calculate ECE for this fold
    ece_cv = 0
    for i in range(n_bins):
        mask = (y_pred_proba_cv >= bin_edges[i]) & (y_pred_proba_cv < bin_edges[i+1])
        if mask.sum() > 0:
            bin_acc = y_test_cv[mask].mean()
            bin_conf = y_pred_proba_cv[mask].mean()
            ece_cv += np.abs(bin_acc - bin_conf) * mask.sum()

    ece_cv = ece_cv / len(y_test_cv) if len(y_test_cv) > 0 else 0
    ece_scores.append(ece_cv)

    # Calculate Brier for this fold
    brier_cv = brier_score_loss(y_test_cv, y_pred_proba_cv)
    brier_scores.append(brier_cv)

    # Calculate MCE for this fold
    mce_cv = 0
    for i in range(n_bins):
        mask = (y_pred_proba_cv >= bin_edges[i]) & (y_pred_proba_cv < bin_edges[i+1])
        if mask.sum() > 0:
            bin_acc = y_test_cv[mask].mean()
            bin_conf = y_pred_proba_cv[mask].mean()
            mce_cv = max(mce_cv, np.abs(bin_acc - bin_conf))

    mce_scores.append(mce_cv)

    print(f"Repeat {repeat+1} (seed={seed}): ECE={ece_cv:.6f}, Brier={brier_cv:.6f}, MCE={mce_cv:.6f}")

print(f"\nStability Summary (across {n_repeats} runs):")
print(f"ECE:   mean={np.mean(ece_scores):.6f}, std={np.std(ece_scores):.6f}, range=[{np.min(ece_scores):.6f}, {np.max(ece_scores):.6f}]")
print(f"Brier: mean={np.mean(brier_scores):.6f}, std={np.std(brier_scores):.6f}, range=[{np.min(brier_scores):.6f}, {np.max(brier_scores):.6f}]")
print(f"MCE:   mean={np.mean(mce_scores):.6f}, std={np.std(mce_scores):.6f}, range=[{np.min(mce_scores):.6f}, {np.max(mce_scores):.6f}]")

print("\n" + "=" * 60)
print("CALIBRATION ASSESSMENT")
print("=" * 60)

# Thresholds for well-calibrated models (common benchmarks)
print("\nCommunly accepted calibration thresholds:")
print("ECE < 0.05: Very well-calibrated")
print("ECE < 0.10: Well-calibrated")
print("ECE >= 0.10: Poorly-calibrated")

mean_ece = np.mean(ece_scores)
if mean_ece < 0.05:
    calibration_status = "Very well-calibrated"
elif mean_ece < 0.10:
    calibration_status = "Well-calibrated"
else:
    calibration_status = "Poorly-calibrated"

print(f"\nModel assessment: {calibration_status} (ECE={mean_ece:.6f})")

# Direction assessment
direction = "model is well-calibrated" if mean_ece < 0.10 else "model is poorly-calibrated"

print("\n" + "=" * 60)
print("FINDINGS SUMMARY")
print("=" * 60)
print(f"Primary metric: Expected Calibration Error (ECE)")
print(f"Primary metric value (mean across repeats): {mean_ece:.6f}")
print(f"Direction: {direction}")
print(f"Conclusion: ECE of {mean_ece:.4f} indicates the model is {'well-calibrated' if mean_ece < 0.10 else 'poorly-calibrated'}")
