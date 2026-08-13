import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_validate, RepeatedStratifiedKFold
from sklearn.metrics import f1_score, make_scorer
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df['class'].value_counts())
print("\nFeatures:", df.columns.tolist())

# Prepare features and target
X = df.drop('class', axis=1)
y = df['class'].map({'<=50K': 0, '>50K': 1})

print("\nClass distribution:")
print(f"Negative (<=50K): {(y == 0).sum()}")
print(f"Positive (>50K): {(y == 1).sum()}")
print(f"Minority class proportion: {(y == 1).sum() / len(y):.3f}")

# Handle missing values and encode categorical features
# Replace '?' with NaN
X = X.replace('?', np.nan)
X = X.replace('', np.nan)

# Drop missing values (or fill with mode)
X = X.fillna(X.mode().iloc[0])

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numerical columns: {numerical_cols}")

# Encode categorical features
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

X = X.astype('float64')
print(f"\nFinal X shape: {X.shape}")

# ======================================================================
# MAIN ANALYSIS: Compare F1 score with and without SMOTE
# ======================================================================

# Random state for reproducibility
RANDOM_STATE = 42

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train minority class: {(y_train == 1).sum()}")
print(f"Test minority class: {(y_test == 1).sum()}")

# ======================================================================
# Model 1: Random Forest without SMOTE
# ======================================================================
print("\n" + "="*70)
print("MODEL 1: Random Forest WITHOUT SMOTE")
print("="*70)

rf_no_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_no_smote.fit(X_train, y_train)
y_pred_no_smote = rf_no_smote.predict(X_test)

f1_no_smote = f1_score(y_test, y_pred_no_smote, pos_label=1)
print(f"Minority-class (>50K) F1 score: {f1_no_smote:.6f}")

# ======================================================================
# Model 2: Random Forest with SMOTE on training data
# ======================================================================
print("\n" + "="*70)
print("MODEL 2: Random Forest WITH SMOTE on training data")
print("="*70)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Train set after SMOTE:")
print(f"  Total samples: {X_train_smote.shape[0]}")
print(f"  Minority class: {(y_train_smote == 1).sum()}")
print(f"  Majority class: {(y_train_smote == 0).sum()}")

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_smote, y_train_smote)
y_pred_smote = rf_smote.predict(X_test)

f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)
print(f"Minority-class (>50K) F1 score: {f1_smote:.6f}")

# ======================================================================
# PRIMARY RESULT
# ======================================================================
f1_difference = f1_smote - f1_no_smote
print("\n" + "="*70)
print("PRIMARY RESULT")
print("="*70)
print(f"F1 score WITHOUT SMOTE: {f1_no_smote:.6f}")
print(f"F1 score WITH SMOTE:    {f1_smote:.6f}")
print(f"Difference (SMOTE - No SMOTE): {f1_difference:.6f}")
print(f"Does SMOTE change F1 by more than 0.02? {abs(f1_difference) > 0.02}")

# ======================================================================
# VALIDATION: Repeated Stratified K-Fold Cross-Validation
# ======================================================================
print("\n" + "="*70)
print("VALIDATION: Repeated Stratified K-Fold Cross-Validation")
print("="*70)

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

f1_scores_no_smote = []
f1_scores_smote = []
differences = []

for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
    X_train_cv = X.iloc[train_idx]
    X_test_cv = X.iloc[test_idx]
    y_train_cv = y.iloc[train_idx]
    y_test_cv = y.iloc[test_idx]

    # Model without SMOTE
    rf_cv_no_smote = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_cv_no_smote.fit(X_train_cv, y_train_cv)
    y_pred_cv_no_smote = rf_cv_no_smote.predict(X_test_cv)
    f1_cv_no_smote = f1_score(y_test_cv, y_pred_cv_no_smote, pos_label=1)
    f1_scores_no_smote.append(f1_cv_no_smote)

    # Model with SMOTE
    smote_cv = SMOTE(random_state=RANDOM_STATE)
    X_train_cv_smote, y_train_cv_smote = smote_cv.fit_resample(X_train_cv, y_train_cv)
    rf_cv_smote = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_cv_smote.fit(X_train_cv_smote, y_train_cv_smote)
    y_pred_cv_smote = rf_cv_smote.predict(X_test_cv)
    f1_cv_smote = f1_score(y_test_cv, y_pred_cv_smote, pos_label=1)
    f1_scores_smote.append(f1_cv_smote)

    diff = f1_cv_smote - f1_cv_no_smote
    differences.append(diff)

f1_no_smote_mean = np.mean(f1_scores_no_smote)
f1_smote_mean = np.mean(f1_scores_smote)
f1_no_smote_std = np.std(f1_scores_no_smote)
f1_smote_std = np.std(f1_scores_smote)
diff_mean = np.mean(differences)
diff_std = np.std(differences)

print(f"\nWithout SMOTE (5x5 CV):")
print(f"  Mean F1: {f1_no_smote_mean:.6f}")
print(f"  Std:     {f1_no_smote_std:.6f}")
print(f"  Min:     {np.min(f1_scores_no_smote):.6f}")
print(f"  Max:     {np.max(f1_scores_no_smote):.6f}")

print(f"\nWith SMOTE (5x5 CV):")
print(f"  Mean F1: {f1_smote_mean:.6f}")
print(f"  Std:     {f1_smote_std:.6f}")
print(f"  Min:     {np.min(f1_scores_smote):.6f}")
print(f"  Max:     {np.max(f1_scores_smote):.6f}")

print(f"\nDifference (SMOTE - No SMOTE) across CV folds:")
print(f"  Mean:    {diff_mean:.6f}")
print(f"  Std:     {diff_std:.6f}")
print(f"  Min:     {np.min(differences):.6f}")
print(f"  Max:     {np.max(differences):.6f}")
print(f"  95% CI:  [{diff_mean - 1.96*diff_std:.6f}, {diff_mean + 1.96*diff_std:.6f}]")

print(f"\nHolds up across CV folds? {abs(diff_mean) > 0.02}")
print(f"Consistent direction? All folds > 0.02 threshold: {all(abs(d) > 0.02 for d in differences)}")
print(f"Consistent direction? All positive changes: {all(d > 0 for d in differences)}")

# ======================================================================
# FINAL SUMMARY
# ======================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)
print(f"\nOriginal train-test split finding:")
print(f"  Change in F1 score: {f1_difference:.6f}")
print(f"  Exceeds 0.02 threshold: {abs(f1_difference) > 0.02}")

print(f"\nCross-validation validation (5x5 repeated CV):")
print(f"  Mean change in F1 score: {diff_mean:.6f}")
print(f"  Std dev: {diff_std:.6f}")
print(f"  Exceeds 0.02 threshold on average: {abs(diff_mean) > 0.02}")

if abs(f1_difference) > 0.02 and abs(diff_mean) > 0.02:
    finding = "YES - SMOTE DOES change F1 by more than 0.02"
    direction = f"SMOTE improves F1" if f1_difference > 0 else "SMOTE decreases F1"
else:
    finding = "NO - SMOTE does NOT change F1 by more than 0.02"
    direction = f"Change is {f1_difference:.6f} (within 0.02 threshold)"

print(f"\nConclusion: {finding}")
print(f"Direction: {direction}")
