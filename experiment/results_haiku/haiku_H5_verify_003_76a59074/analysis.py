"""
Analysis of SMOTE oversampling effect on minority-class F1 score
Research Question: Does SMOTE oversampling change the minority-class (>50K) F1 score
by more than 0.02 compared to no resampling, holding classifier fixed as default RandomForest?
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Load data
print("=" * 80)
print("LOADING DATA")
print("=" * 80)
df = pd.read_csv('adult_income.csv')
print(f"Data shape: {df.shape}")
print(f"\nClass distribution:\n{df['class'].value_counts()}")
print(f"\nClass proportions:\n{df['class'].value_counts(normalize=True)}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Data preprocessing
print("\n" + "=" * 80)
print("PREPROCESSING")
print("=" * 80)

# Remove rows with missing target
df = df[df['class'].notna()].copy()

# Handle missing values in features
# For categorical columns with missing values, fill with 'Unknown'
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('class')  # Remove target

for col in categorical_cols:
    df[col] = df[col].fillna('Unknown')

print(f"Data shape after removing missing targets: {df.shape}")
print(f"Remaining missing values:\n{df.isnull().sum()}")

# Encode categorical variables
print("\nEncoding categorical variables...")
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    le_dict[col] = le

# Encode target variable
le_target = LabelEncoder()
df['class'] = le_target.fit_transform(df['class'])  # <=50K -> 0, >50K -> 1

print(f"Target encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
print(f"Minority class (>50K) is encoded as: 1")

# Prepare features and target
X = df.drop('class', axis=1)
y = df['class']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Minority class (1) count: {(y == 1).sum()}")
print(f"Majority class (0) count: {(y == 0).sum()}")

# Train-test split
print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT (80/20)")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set class distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test set class distribution:\n{pd.Series(y_test).value_counts()}")

# Baseline: Train RF without SMOTE
print("\n" + "=" * 80)
print("BASELINE: RANDOM FOREST WITHOUT SMOTE")
print("=" * 80)

rf_baseline = RandomForestClassifier(random_state=RANDOM_STATE)
rf_baseline.fit(X_train, y_train)

y_pred_baseline = rf_baseline.predict(X_test)
f1_baseline = f1_score(y_test, y_pred_baseline, pos_label=1)

print(f"F1 score (minority class >50K) WITHOUT SMOTE: {f1_baseline:.6f}")
print(f"\nClassification Report (Baseline):")
print(classification_report(y_test, y_pred_baseline, target_names=['<=50K', '>50K']))

# Apply SMOTE
print("\n" + "=" * 80)
print("APPLYING SMOTE OVERSAMPLING")
print("=" * 80)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set size after SMOTE: {X_train_smote.shape[0]}")
print(f"Training set class distribution after SMOTE:\n{pd.Series(y_train_smote).value_counts()}")
print(f"Class balance ratio after SMOTE: {pd.Series(y_train_smote).value_counts()[1] / pd.Series(y_train_smote).value_counts()[0]:.4f}")

# Train RF with SMOTE
print("\n" + "=" * 80)
print("RANDOM FOREST WITH SMOTE")
print("=" * 80)

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = rf_smote.predict(X_test)
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)

print(f"F1 score (minority class >50K) WITH SMOTE: {f1_smote:.6f}")
print(f"\nClassification Report (SMOTE):")
print(classification_report(y_test, y_pred_smote, target_names=['<=50K', '>50K']))

# Compare
print("\n" + "=" * 80)
print("COMPARISON")
print("=" * 80)

f1_difference = abs(f1_smote - f1_baseline)
threshold = 0.02

print(f"F1 score WITHOUT SMOTE: {f1_baseline:.6f}")
print(f"F1 score WITH SMOTE:    {f1_smote:.6f}")
print(f"Absolute difference:    {f1_difference:.6f}")
print(f"Threshold:              {threshold:.6f}")
print(f"\nDoes SMOTE change F1 score by MORE than {threshold}? {f1_difference > threshold}")

# Validation: Repeated 5-fold stratified cross-validation
print("\n" + "=" * 80)
print("VALIDATION: REPEATED STRATIFIED K-FOLD CROSS-VALIDATION")
print("=" * 80)
print("5 repeats of 5-fold stratified cross-validation with different random seeds")

f1_scores_baseline = []
f1_scores_smote = []

for repeat in range(5):
    seed = RANDOM_STATE + repeat
    np.random.seed(seed)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    f1_baseline_fold = []
    f1_smote_fold = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        # Baseline
        rf_b = RandomForestClassifier(random_state=seed)
        rf_b.fit(X_tr, y_tr)
        y_pred_b = rf_b.predict(X_val)
        f1_b = f1_score(y_val, y_pred_b, pos_label=1)
        f1_baseline_fold.append(f1_b)

        # With SMOTE
        smote_cv = SMOTE(random_state=seed)
        X_tr_smote, y_tr_smote = smote_cv.fit_resample(X_tr, y_tr)
        rf_s = RandomForestClassifier(random_state=seed)
        rf_s.fit(X_tr_smote, y_tr_smote)
        y_pred_s = rf_s.predict(X_val)
        f1_s = f1_score(y_val, y_pred_s, pos_label=1)
        f1_smote_fold.append(f1_s)

    f1_scores_baseline.extend(f1_baseline_fold)
    f1_scores_smote.extend(f1_smote_fold)

    print(f"Repeat {repeat+1}: Baseline mean={np.mean(f1_baseline_fold):.6f}, SMOTE mean={np.mean(f1_smote_fold):.6f}")

f1_baseline_cv = np.array(f1_scores_baseline)
f1_smote_cv = np.array(f1_scores_smote)

print(f"\n" + "-" * 80)
print("CROSS-VALIDATION RESULTS (25 folds total)")
print("-" * 80)
print(f"Baseline F1 scores - Mean: {f1_baseline_cv.mean():.6f}, Std: {f1_baseline_cv.std():.6f}")
print(f"  95% CI: [{f1_baseline_cv.mean() - 1.96*f1_baseline_cv.std():.6f}, {f1_baseline_cv.mean() + 1.96*f1_baseline_cv.std():.6f}]")
print(f"\nSMOTE F1 scores - Mean: {f1_smote_cv.mean():.6f}, Std: {f1_smote_cv.std():.6f}")
print(f"  95% CI: [{f1_smote_cv.mean() - 1.96*f1_smote_cv.std():.6f}, {f1_smote_cv.mean() + 1.96*f1_smote_cv.std():.6f}]")

mean_difference_cv = f1_smote_cv.mean() - f1_baseline_cv.mean()
print(f"\nMean F1 difference (SMOTE - Baseline): {mean_difference_cv:.6f}")
print(f"Absolute difference: {abs(mean_difference_cv):.6f}")
print(f"Does cross-validation confirm difference > {threshold}? {abs(mean_difference_cv) > threshold}")

# Final assessment
print("\n" + "=" * 80)
print("FINAL ASSESSMENT")
print("=" * 80)

answer = f1_difference > threshold
cv_answer = abs(mean_difference_cv) > threshold

print(f"Test set F1 difference: {f1_difference:.6f} {'>' if f1_difference > threshold else '<='} {threshold}")
print(f"Cross-validation confirms: {cv_answer}")
print(f"\nRESEARCH QUESTION ANSWER:")
print(f"Does SMOTE change F1 score by MORE than {threshold}? {answer}")

# Store results for JSON output
results = {
    'test_f1_baseline': float(f1_baseline),
    'test_f1_smote': float(f1_smote),
    'test_f1_difference': float(f1_difference),
    'cv_f1_baseline_mean': float(f1_baseline_cv.mean()),
    'cv_f1_baseline_std': float(f1_baseline_cv.std()),
    'cv_f1_smote_mean': float(f1_smote_cv.mean()),
    'cv_f1_smote_std': float(f1_smote_cv.std()),
    'cv_mean_difference': float(mean_difference_cv),
    'answer': answer,
    'cv_confirms': cv_answer,
    'threshold': threshold
}

print(f"\n{results}")
