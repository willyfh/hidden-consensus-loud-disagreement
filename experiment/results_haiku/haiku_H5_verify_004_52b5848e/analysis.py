import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nColumn names and types:")
print(df.dtypes)
print(f"\nTarget distribution:")
print(df['class'].value_counts())

# Data exploration
print(f"\nMissing values:")
print(df.isnull().sum())

# Preprocessing
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

# Handle missing values (represented as '?')
print("\nChecking for missing values marked as '?'...")
for col in df.columns:
    if df[col].dtype == 'object':
        missing_count = (df[col] == ' ?').sum()
        if missing_count > 0:
            print(f"  {col}: {missing_count} missing values")
            # Drop rows with missing values
            df = df[df[col] != ' ?']

print(f"\nDataset shape after removing rows with '?': {df.shape}")

# Separate target and features
y = df['class'].copy()
X = df.drop('class', axis=1).copy()

# Encode target variable
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print(f"\nTarget encoding: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")
print(f"Class distribution after encoding:")
print(f"  Class 0 (<=50K): {(y_encoded == 0).sum()}")
print(f"  Class 1 (>50K): {(y_encoded == 1).sum()}")

# Encode categorical features
print("\nEncoding categorical features...")
categorical_cols = X.select_dtypes(include='object').columns.tolist()
print(f"Categorical columns: {categorical_cols}")

le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

print(f"\nFeature matrix shape: {X.shape}")
print(f"Feature types after encoding:\n{X.dtypes}")

# Train-test split
print("\n" + "="*60)
print("TRAIN-TEST SPLIT AND MODEL TRAINING")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train set class distribution:")
print(f"  Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
print(f"Test set class distribution:")
print(f"  Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()}")

# Model 1: No resampling
print("\n" + "-"*60)
print("Model 1: Random Forest WITHOUT SMOTE")
print("-"*60)

rf_no_smote = RandomForestClassifier()
rf_no_smote.fit(X_train, y_train)

y_pred_no_smote = rf_no_smote.predict(X_test)
f1_no_smote = f1_score(y_test, y_pred_no_smote, pos_label=1)

print(f"\nF1 Score (minority class >50K): {f1_no_smote:.6f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_no_smote, target_names=['<=50K', '>50K']))

# Model 2: With SMOTE
print("\n" + "-"*60)
print("Model 2: Random Forest WITH SMOTE")
print("-"*60)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Train set size after SMOTE: {X_train_smote.shape[0]}")
print(f"Train set class distribution after SMOTE:")
print(f"  Class 0: {(y_train_smote == 0).sum()}, Class 1: {(y_train_smote == 1).sum()}")

rf_smote = RandomForestClassifier()
rf_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = rf_smote.predict(X_test)
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)

print(f"\nF1 Score (minority class >50K): {f1_smote:.6f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_smote, target_names=['<=50K', '>50K']))

# Comparison
print("\n" + "="*60)
print("COMPARISON")
print("="*60)

f1_difference = f1_smote - f1_no_smote
print(f"\nF1 Score WITHOUT SMOTE: {f1_no_smote:.6f}")
print(f"F1 Score WITH SMOTE:    {f1_smote:.6f}")
print(f"Difference (SMOTE - No SMOTE): {f1_difference:.6f}")
print(f"\nDoes the difference exceed 0.02? {abs(f1_difference) > 0.02}")
print(f"Absolute difference > 0.02: {abs(f1_difference) > 0.02}")

# Validation with cross-validation
print("\n" + "="*60)
print("VALIDATION WITH CROSS-VALIDATION")
print("="*60)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_no_smote = []
cv_results_smote = []
cv_differences = []

print("\nRunning 5-Fold Cross-Validation with 5 different random seeds...")

for seed in [42, 123, 456, 789, 999]:
    cv_seed = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(cv_seed.split(X, y_encoded)):
        X_cv_train, X_cv_test = X.iloc[train_idx], X.iloc[test_idx]
        y_cv_train, y_cv_test = y_encoded[train_idx], y_encoded[test_idx]

        # Without SMOTE
        rf_cv_no_smote = RandomForestClassifier(random_state=seed)
        rf_cv_no_smote.fit(X_cv_train, y_cv_train)
        y_cv_pred_no_smote = rf_cv_no_smote.predict(X_cv_test)
        f1_cv_no_smote = f1_score(y_cv_test, y_cv_pred_no_smote, pos_label=1)
        cv_results_no_smote.append(f1_cv_no_smote)

        # With SMOTE
        smote_cv = SMOTE(random_state=seed)
        X_cv_train_smote, y_cv_train_smote = smote_cv.fit_resample(X_cv_train, y_cv_train)
        rf_cv_smote = RandomForestClassifier(random_state=seed)
        rf_cv_smote.fit(X_cv_train_smote, y_cv_train_smote)
        y_cv_pred_smote = rf_cv_smote.predict(X_cv_test)
        f1_cv_smote = f1_score(y_cv_test, y_cv_pred_smote, pos_label=1)
        cv_results_smote.append(f1_cv_smote)

        cv_differences.append(f1_cv_smote - f1_cv_no_smote)

cv_results_no_smote = np.array(cv_results_no_smote)
cv_results_smote = np.array(cv_results_smote)
cv_differences = np.array(cv_differences)

print(f"\nCross-Validation Results (25 folds: 5 seeds × 5 folds):")
print(f"\nWithout SMOTE:")
print(f"  Mean F1: {cv_results_no_smote.mean():.6f}")
print(f"  Std Dev: {cv_results_no_smote.std():.6f}")
print(f"  Min: {cv_results_no_smote.min():.6f}, Max: {cv_results_no_smote.max():.6f}")

print(f"\nWith SMOTE:")
print(f"  Mean F1: {cv_results_smote.mean():.6f}")
print(f"  Std Dev: {cv_results_smote.std():.6f}")
print(f"  Min: {cv_results_smote.min():.6f}, Max: {cv_results_smote.max():.6f}")

print(f"\nF1 Difference (SMOTE - No SMOTE):")
print(f"  Mean: {cv_differences.mean():.6f}")
print(f"  Std Dev: {cv_differences.std():.6f}")
print(f"  Min: {cv_differences.min():.6f}, Max: {cv_differences.max():.6f}")
print(f"  95% CI: [{cv_differences.mean() - 1.96*cv_differences.std():.6f}, {cv_differences.mean() + 1.96*cv_differences.std():.6f}]")

# Count how many folds show difference > 0.02
num_folds_diff_gt_002 = (np.abs(cv_differences) > 0.02).sum()
print(f"\nNumber of folds with |difference| > 0.02: {num_folds_diff_gt_002} / {len(cv_differences)}")

# Final conclusions
print("\n" + "="*60)
print("FINAL FINDINGS")
print("="*60)

print(f"\nPrimary Finding from Train-Test Split:")
print(f"  F1 without SMOTE: {f1_no_smote:.6f}")
print(f"  F1 with SMOTE:    {f1_smote:.6f}")
print(f"  Difference:       {f1_difference:.6f}")
print(f"  |Difference| > 0.02? {abs(f1_difference) > 0.02}")

print(f"\nStability Check (Cross-Validation):")
print(f"  Mean difference across {len(cv_differences)} folds: {cv_differences.mean():.6f}")
print(f"  95% CI: [{cv_differences.mean() - 1.96*cv_differences.std():.6f}, {cv_differences.mean() + 1.96*cv_differences.std():.6f}]")
print(f"  Finding holds up (mean |diff| > 0.02)? {abs(cv_differences.mean()) > 0.02}")

# Final determination
answer = abs(f1_difference) > 0.02
answer_cv = abs(cv_differences.mean()) > 0.02

print(f"\n{'*'*60}")
print(f"ANSWER TO RESEARCH QUESTION:")
print(f"Does SMOTE change F1 score by MORE than 0.02?")
print(f"  Based on Train-Test: {answer}")
print(f"  Based on CV (mean):  {answer_cv}")
print(f"{'*'*60}")
