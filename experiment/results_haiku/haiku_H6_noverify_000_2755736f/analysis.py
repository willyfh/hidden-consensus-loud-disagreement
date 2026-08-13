import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget distribution:")
print(df['class'].value_counts())

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target: <=50K = 0, >50K = 1
y_encoded = (y == '>50K').astype(int)

print("\nEncoded target distribution:")
print(pd.Series(y_encoded).value_counts())
print(f"Positive class rate: {y_encoded.mean():.4f}")

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# Handle missing values in categorical columns
for col in categorical_cols:
    if X[col].isnull().any():
        print(f"Missing in {col}: {X[col].isnull().sum()}")
        X[col].fillna(X[col].mode()[0], inplace=True)

# Handle missing values in numeric columns
for col in numeric_cols:
    if X[col].isnull().any():
        print(f"Missing in {col}: {X[col].isnull().sum()}")
        X[col].fillna(X[col].median(), inplace=True)

# Encode categorical variables
le_dict = {}
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

print("\nEncoded data shape:", X_encoded.shape)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"Train positive rate: {y_train.mean():.4f}")
print(f"Test positive rate: {y_test.mean():.4f}")

# Train a logistic regression model
print("\n=== Training Logistic Regression ===")
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train, y_train)
y_pred_proba_lr = lr.predict_proba(X_test)[:, 1]

# Train a random forest model
print("=== Training Random Forest ===")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_proba_rf = rf.predict_proba(X_test)[:, 1]

# Compute calibration curves
print("\n=== Computing Calibration Metrics ===")

def compute_calibration_metrics(y_true, y_pred_proba, n_bins=10):
    """Compute calibration curve and expected calibration error."""
    prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=n_bins, strategy='uniform')

    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(prob_true - prob_pred))

    # Maximum Calibration Error (MCE)
    mce = np.max(np.abs(prob_true - prob_pred))

    # Brier Score
    brier = np.mean((y_pred_proba - y_true) ** 2)

    return {
        'ece': ece,
        'mce': mce,
        'brier': brier,
        'prob_true': prob_true,
        'prob_pred': prob_pred
    }

metrics_lr = compute_calibration_metrics(y_test, y_pred_proba_lr)
metrics_rf = compute_calibration_metrics(y_test, y_pred_proba_rf)

print(f"\nLogistic Regression:")
print(f"  Expected Calibration Error (ECE): {metrics_lr['ece']:.4f}")
print(f"  Maximum Calibration Error (MCE): {metrics_lr['mce']:.4f}")
print(f"  Brier Score: {metrics_lr['brier']:.4f}")

print(f"\nRandom Forest:")
print(f"  Expected Calibration Error (ECE): {metrics_rf['ece']:.4f}")
print(f"  Maximum Calibration Error (MCE): {metrics_rf['mce']:.4f}")
print(f"  Brier Score: {metrics_rf['brier']:.4f}")

# Additional analysis: check calibration by binning predictions
print("\n=== Calibration Analysis (by probability bins) ===")

def analyze_calibration_by_bins(y_true, y_pred_proba, n_bins=10):
    """Analyze calibration by binning predictions."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bins = np.digitize(y_pred_proba, bin_edges) - 1

    results = []
    for i in range(n_bins):
        mask = bins == i
        if mask.sum() > 0:
            mean_pred = y_pred_proba[mask].mean()
            mean_actual = y_true[mask].mean()
            n_samples = mask.sum()
            results.append({
                'bin': i,
                'n_samples': n_samples,
                'mean_pred_prob': mean_pred,
                'mean_actual_freq': mean_actual,
                'calibration_error': abs(mean_pred - mean_actual)
            })
    return pd.DataFrame(results)

print("\nLogistic Regression:")
lr_bins = analyze_calibration_by_bins(y_test, y_pred_proba_lr)
print(lr_bins)

print("\nRandom Forest:")
rf_bins = analyze_calibration_by_bins(y_test, y_pred_proba_rf)
print(rf_bins)

# Create calibration plot visualization
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LR calibration plot
axes[0].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
axes[0].plot(metrics_lr['prob_pred'], metrics_lr['prob_true'], 's-', label='Logistic Regression', markersize=8)
axes[0].set_xlabel('Mean predicted probability', fontsize=11)
axes[0].set_ylabel('Fraction of positives', fontsize=11)
axes[0].set_title('Calibration Plot: Logistic Regression', fontsize=12)
axes[0].legend(loc='lower right')
axes[0].set_xlim([-0.05, 1.05])
axes[0].set_ylim([-0.05, 1.05])
axes[0].grid(alpha=0.3)

# RF calibration plot
axes[1].plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=2)
axes[1].plot(metrics_rf['prob_pred'], metrics_rf['prob_true'], 's-', label='Random Forest', markersize=8, color='orange')
axes[1].set_xlabel('Mean predicted probability', fontsize=11)
axes[1].set_ylabel('Fraction of positives', fontsize=11)
axes[1].set_title('Calibration Plot: Random Forest', fontsize=12)
axes[1].legend(loc='lower right')
axes[1].set_xlim([-0.05, 1.05])
axes[1].set_ylim([-0.05, 1.05])
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('calibration_plots.png', dpi=100, bbox_inches='tight')
print("\nCalibration plots saved to calibration_plots.png")

# Summary
print("\n" + "="*60)
print("CALIBRATION ASSESSMENT SUMMARY")
print("="*60)

# Interpretation thresholds (ECE)
def interpret_calibration(ece):
    if ece < 0.05:
        return "Excellent (well-calibrated)"
    elif ece < 0.10:
        return "Good"
    elif ece < 0.15:
        return "Fair"
    elif ece < 0.20:
        return "Poor"
    else:
        return "Very poor (miscalibrated)"

print(f"\nLogistic Regression: ECE = {metrics_lr['ece']:.4f} → {interpret_calibration(metrics_lr['ece'])}")
print(f"Random Forest: ECE = {metrics_rf['ece']:.4f} → {interpret_calibration(metrics_rf['ece'])}")

# Determine overall finding
avg_ece = (metrics_lr['ece'] + metrics_rf['ece']) / 2
better_model_ece = min(metrics_lr['ece'], metrics_rf['ece'])
better_model_name = 'Logistic Regression' if metrics_lr['ece'] < metrics_rf['ece'] else 'Random Forest'

print(f"\nBetter calibrated model: {better_model_name} (ECE = {better_model_ece:.4f})")
print(f"\nConclusion: The {better_model_name} model shows {'good' if better_model_ece < 0.10 else 'fair to poor'} calibration.")
print(f"An ECE of {better_model_ece:.4f} indicates predicted probabilities {'align well' if better_model_ece < 0.10 else 'deviate moderately'} with observed frequencies.")
