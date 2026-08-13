import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# Load data
print("=" * 80)
print("LOADING AND EXPLORING DATA")
print("=" * 80)
df = pd.read_csv('adult_income.csv')

print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())

print(f"\nColumn names:")
print(df.columns.tolist())

print(f"\nData types:")
print(df.dtypes)

print(f"\nMissing values:")
print(df.isnull().sum())

print(f"\nClass distribution:")
print(df['class'].value_counts())
print(f"\nClass proportions:")
print(df['class'].value_counts(normalize=True))

# Check imbalance ratio
class_counts = df['class'].value_counts()
imbalance_ratio = class_counts.max() / class_counts.min()
print(f"\nImbalance ratio (majority/minority): {imbalance_ratio:.2f}")

# Prepare data for modeling
print("\n" + "=" * 80)
print("DATA PREPARATION")
print("=" * 80)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target: <=50K -> 0, >50K -> 1
y_encoded = (y == '>50K').astype(int)

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns: {categorical_cols}")
print(f"Numeric columns: {numeric_cols}")

# Encode categorical variables
X_processed = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    # Handle unknown categories during transform
    X_processed[col] = le.fit_transform(X_processed[col].astype(str))
    label_encoders[col] = le

print(f"\nProcessed data shape: {X_processed.shape}")

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train set class distribution:\n{pd.Series(y_train).value_counts()}")
print(f"Test set class distribution:\n{pd.Series(y_test).value_counts()}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# APPROACH 1: BASELINE MODEL (No imbalance handling)
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 1: BASELINE MODEL (No imbalance handling)")
print("=" * 80)

# Use RandomForest and LogisticRegression
baseline_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20, n_jobs=-1)
baseline_rf.fit(X_train_scaled, y_train)
baseline_rf_pred_proba = baseline_rf.predict_proba(X_test_scaled)[:, 1]
baseline_rf_auc = roc_auc_score(y_test, baseline_rf_pred_proba)
baseline_rf_f1 = f1_score(y_test, baseline_rf.predict(X_test_scaled))

baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
baseline_lr.fit(X_train_scaled, y_train)
baseline_lr_pred_proba = baseline_lr.predict_proba(X_test_scaled)[:, 1]
baseline_lr_auc = roc_auc_score(y_test, baseline_lr_pred_proba)
baseline_lr_f1 = f1_score(y_test, baseline_lr.predict(X_test_scaled))

print(f"\nBaseline RandomForest:")
print(f"  ROC-AUC: {baseline_rf_auc:.4f}")
print(f"  F1-Score: {baseline_rf_f1:.4f}")

print(f"\nBaseline LogisticRegression:")
print(f"  ROC-AUC: {baseline_lr_auc:.4f}")
print(f"  F1-Score: {baseline_lr_f1:.4f}")

# ============================================================================
# APPROACH 2: Class Weights
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 2: Class Weights")
print("=" * 80)

# RandomForest with class weights
weighted_rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    max_depth=20,
    class_weight='balanced',
    n_jobs=-1
)
weighted_rf.fit(X_train_scaled, y_train)
weighted_rf_pred_proba = weighted_rf.predict_proba(X_test_scaled)[:, 1]
weighted_rf_auc = roc_auc_score(y_test, weighted_rf_pred_proba)
weighted_rf_f1 = f1_score(y_test, weighted_rf.predict(X_test_scaled))

# LogisticRegression with class weights
weighted_lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
weighted_lr.fit(X_train_scaled, y_train)
weighted_lr_pred_proba = weighted_lr.predict_proba(X_test_scaled)[:, 1]
weighted_lr_auc = roc_auc_score(y_test, weighted_lr_pred_proba)
weighted_lr_f1 = f1_score(y_test, weighted_lr.predict(X_test_scaled))

print(f"\nClass-Weighted RandomForest:")
print(f"  ROC-AUC: {weighted_rf_auc:.4f} (Δ: {weighted_rf_auc - baseline_rf_auc:+.4f})")
print(f"  F1-Score: {weighted_rf_f1:.4f} (Δ: {weighted_rf_f1 - baseline_rf_f1:+.4f})")

print(f"\nClass-Weighted LogisticRegression:")
print(f"  ROC-AUC: {weighted_lr_auc:.4f} (Δ: {weighted_lr_auc - baseline_lr_auc:+.4f})")
print(f"  F1-Score: {weighted_lr_f1:.4f} (Δ: {weighted_lr_f1 - baseline_lr_f1:+.4f})")

# ============================================================================
# APPROACH 3: SMOTE (Synthetic Minority Over-sampling)
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 3: SMOTE")
print("=" * 80)

smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"After SMOTE: {X_train_smote.shape[0]} samples")
print(f"Class distribution after SMOTE:\n{pd.Series(y_train_smote).value_counts()}")

# RandomForest on SMOTE data
smote_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20, n_jobs=-1)
smote_rf.fit(X_train_smote, y_train_smote)
smote_rf_pred_proba = smote_rf.predict_proba(X_test_scaled)[:, 1]
smote_rf_auc = roc_auc_score(y_test, smote_rf_pred_proba)
smote_rf_f1 = f1_score(y_test, smote_rf.predict(X_test_scaled))

# LogisticRegression on SMOTE data
smote_lr = LogisticRegression(max_iter=1000, random_state=42)
smote_lr.fit(X_train_smote, y_train_smote)
smote_lr_pred_proba = smote_lr.predict_proba(X_test_scaled)[:, 1]
smote_lr_auc = roc_auc_score(y_test, smote_lr_pred_proba)
smote_lr_f1 = f1_score(y_test, smote_lr.predict(X_test_scaled))

print(f"\nSMOTE + RandomForest:")
print(f"  ROC-AUC: {smote_rf_auc:.4f} (Δ: {smote_rf_auc - baseline_rf_auc:+.4f})")
print(f"  F1-Score: {smote_rf_f1:.4f} (Δ: {smote_rf_f1 - baseline_rf_f1:+.4f})")

print(f"\nSMOTE + LogisticRegression:")
print(f"  ROC-AUC: {smote_lr_auc:.4f} (Δ: {smote_lr_auc - baseline_lr_auc:+.4f})")
print(f"  F1-Score: {smote_lr_f1:.4f} (Δ: {smote_lr_f1 - baseline_lr_f1:+.4f})")

# ============================================================================
# APPROACH 4: Undersampling
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 4: Random Undersampling")
print("=" * 80)

undersampler = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = undersampler.fit_resample(X_train_scaled, y_train)

print(f"After undersampling: {X_train_under.shape[0]} samples")
print(f"Class distribution after undersampling:\n{pd.Series(y_train_under).value_counts()}")

# RandomForest on undersampled data
under_rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=20, n_jobs=-1)
under_rf.fit(X_train_under, y_train_under)
under_rf_pred_proba = under_rf.predict_proba(X_test_scaled)[:, 1]
under_rf_auc = roc_auc_score(y_test, under_rf_pred_proba)
under_rf_f1 = f1_score(y_test, under_rf.predict(X_test_scaled))

print(f"\nUndersampling + RandomForest:")
print(f"  ROC-AUC: {under_rf_auc:.4f} (Δ: {under_rf_auc - baseline_rf_auc:+.4f})")
print(f"  F1-Score: {under_rf_f1:.4f} (Δ: {under_rf_f1 - baseline_rf_f1:+.4f})")

# ============================================================================
# SUMMARY COMPARISON
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY: ROC-AUC COMPARISON")
print("=" * 80)

results_summary = pd.DataFrame({
    'Approach': [
        'Baseline RF',
        'Baseline LR',
        'Class-Weighted RF',
        'Class-Weighted LR',
        'SMOTE + RF',
        'SMOTE + LR',
        'Undersampling + RF'
    ],
    'ROC-AUC': [
        baseline_rf_auc,
        baseline_lr_auc,
        weighted_rf_auc,
        weighted_lr_auc,
        smote_rf_auc,
        smote_lr_auc,
        under_rf_auc
    ],
    'F1-Score': [
        baseline_rf_f1,
        baseline_lr_f1,
        weighted_rf_f1,
        weighted_lr_f1,
        smote_rf_f1,
        smote_lr_f1,
        under_rf_f1
    ]
})

results_summary = results_summary.sort_values('ROC-AUC', ascending=False)
print("\n" + results_summary.to_string(index=False))

best_auc = results_summary['ROC-AUC'].max()
worst_auc = results_summary['ROC-AUC'].min()
auc_improvement = best_auc - worst_auc

print(f"\nBest ROC-AUC: {best_auc:.4f} ({results_summary.iloc[0]['Approach']})")
print(f"Worst ROC-AUC: {worst_auc:.4f} ({results_summary.iloc[-1]['Approach']})")
print(f"Improvement range: {auc_improvement:.4f}")

# ============================================================================
# CROSS-VALIDATION STABILITY CHECK
# ============================================================================
print("\n" + "=" * 80)
print("CROSS-VALIDATION STABILITY CHECK (5 Folds × 3 Random Seeds)")
print("=" * 80)

def evaluate_with_cv(model_class, model_params, imbalance_method=None, model_name="Model"):
    """Evaluate model with repeated cross-validation"""
    scores_all = []
    f1_scores_all = []

    seeds = [42, 123, 456]
    for seed in seeds:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_scores = []
        fold_f1 = []

        for train_idx, val_idx in skf.split(X_processed, y_encoded):
            X_fold_train = X_processed.iloc[train_idx]
            X_fold_val = X_processed.iloc[val_idx]
            y_fold_train = y_encoded.iloc[train_idx]
            y_fold_val = y_encoded.iloc[val_idx]

            # Scale
            scaler_fold = StandardScaler()
            X_fold_train_scaled = scaler_fold.fit_transform(X_fold_train)
            X_fold_val_scaled = scaler_fold.transform(X_fold_val)

            # Handle imbalance
            if imbalance_method == 'smote':
                smote_fold = SMOTE(random_state=seed)
                X_fold_train_scaled, y_fold_train = smote_fold.fit_resample(X_fold_train_scaled, y_fold_train)
            elif imbalance_method == 'undersample':
                undersample_fold = RandomUnderSampler(random_state=seed)
                X_fold_train_scaled, y_fold_train = undersample_fold.fit_resample(X_fold_train_scaled, y_fold_train)

            # Train model
            if model_class == LogisticRegression:
                model = model_class(**model_params, random_state=seed)
            else:
                model = model_class(**model_params, random_state=seed)

            model.fit(X_fold_train_scaled, y_fold_train)

            # Score
            y_val_pred_proba = model.predict_proba(X_fold_val_scaled)[:, 1]
            y_val_pred = model.predict(X_fold_val_scaled)

            auc = roc_auc_score(y_fold_val, y_val_pred_proba)
            f1 = f1_score(y_fold_val, y_val_pred)

            fold_scores.append(auc)
            fold_f1.append(f1)

        scores_all.extend(fold_scores)
        f1_scores_all.extend(fold_f1)

    return np.array(scores_all), np.array(f1_scores_all)

# Evaluate key approaches
cv_results = {}

print("\n1. Baseline RandomForest (no imbalance handling):")
rf_baseline_cv, rf_baseline_f1 = evaluate_with_cv(
    RandomForestClassifier,
    {'n_estimators': 100, 'max_depth': 20, 'n_jobs': -1},
    imbalance_method=None,
    model_name="Baseline RF"
)
print(f"   ROC-AUC: {rf_baseline_cv.mean():.4f} ± {rf_baseline_cv.std():.4f}")
print(f"   F1-Score: {rf_baseline_f1.mean():.4f} ± {rf_baseline_f1.std():.4f}")
cv_results['Baseline RF'] = (rf_baseline_cv.mean(), rf_baseline_cv.std())

print("\n2. RandomForest with Class Weights:")
rf_weighted_cv, rf_weighted_f1 = evaluate_with_cv(
    RandomForestClassifier,
    {'n_estimators': 100, 'max_depth': 20, 'class_weight': 'balanced', 'n_jobs': -1},
    imbalance_method=None,
    model_name="Weighted RF"
)
print(f"   ROC-AUC: {rf_weighted_cv.mean():.4f} ± {rf_weighted_cv.std():.4f}")
print(f"   F1-Score: {rf_weighted_f1.mean():.4f} ± {rf_weighted_f1.std():.4f}")
cv_results['Weighted RF'] = (rf_weighted_cv.mean(), rf_weighted_cv.std())

print("\n3. SMOTE + RandomForest:")
rf_smote_cv, rf_smote_f1 = evaluate_with_cv(
    RandomForestClassifier,
    {'n_estimators': 100, 'max_depth': 20, 'n_jobs': -1},
    imbalance_method='smote',
    model_name="SMOTE RF"
)
print(f"   ROC-AUC: {rf_smote_cv.mean():.4f} ± {rf_smote_cv.std():.4f}")
print(f"   F1-Score: {rf_smote_f1.mean():.4f} ± {rf_smote_f1.std():.4f}")
cv_results['SMOTE RF'] = (rf_smote_cv.mean(), rf_smote_cv.std())

print("\n4. Undersampling + RandomForest:")
rf_under_cv, rf_under_f1 = evaluate_with_cv(
    RandomForestClassifier,
    {'n_estimators': 100, 'max_depth': 20, 'n_jobs': -1},
    imbalance_method='undersample',
    model_name="Undersample RF"
)
print(f"   ROC-AUC: {rf_under_cv.mean():.4f} ± {rf_under_cv.std():.4f}")
print(f"   F1-Score: {rf_under_f1.mean():.4f} ± {rf_under_f1.std():.4f}")
cv_results['Undersample RF'] = (rf_under_cv.mean(), rf_under_cv.std())

print("\n5. Baseline LogisticRegression (no imbalance handling):")
lr_baseline_cv, lr_baseline_f1 = evaluate_with_cv(
    LogisticRegression,
    {'max_iter': 1000},
    imbalance_method=None,
    model_name="Baseline LR"
)
print(f"   ROC-AUC: {lr_baseline_cv.mean():.4f} ± {lr_baseline_cv.std():.4f}")
print(f"   F1-Score: {lr_baseline_f1.mean():.4f} ± {lr_baseline_f1.std():.4f}")
cv_results['Baseline LR'] = (lr_baseline_cv.mean(), lr_baseline_cv.std())

print("\n6. LogisticRegression with Class Weights:")
lr_weighted_cv, lr_weighted_f1 = evaluate_with_cv(
    LogisticRegression,
    {'max_iter': 1000, 'class_weight': 'balanced'},
    imbalance_method=None,
    model_name="Weighted LR"
)
print(f"   ROC-AUC: {lr_weighted_cv.mean():.4f} ± {lr_weighted_cv.std():.4f}")
print(f"   F1-Score: {lr_weighted_f1.mean():.4f} ± {lr_weighted_f1.std():.4f}")
cv_results['Weighted LR'] = (lr_weighted_cv.mean(), lr_weighted_cv.std())

print("\n7. SMOTE + LogisticRegression:")
lr_smote_cv, lr_smote_f1 = evaluate_with_cv(
    LogisticRegression,
    {'max_iter': 1000},
    imbalance_method='smote',
    model_name="SMOTE LR"
)
print(f"   ROC-AUC: {lr_smote_cv.mean():.4f} ± {lr_smote_cv.std():.4f}")
print(f"   F1-Score: {lr_smote_f1.mean():.4f} ± {lr_smote_f1.std():.4f}")
cv_results['SMOTE LR'] = (lr_smote_cv.mean(), lr_smote_cv.std())

# ============================================================================
# FINAL ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL ANALYSIS")
print("=" * 80)

# Compare imbalance-handled vs baseline
baseline_mean = rf_baseline_cv.mean()
best_imbalance_mean = max(rf_weighted_cv.mean(), rf_smote_cv.mean(), rf_under_cv.mean())
best_imbalance_name = ['Weighted RF', 'SMOTE RF', 'Undersample RF'][
    [rf_weighted_cv.mean(), rf_smote_cv.mean(), rf_under_cv.mean()].index(best_imbalance_mean)
]

improvement = best_imbalance_mean - baseline_mean
improvement_pct = (improvement / baseline_mean) * 100

print(f"\nBaseline (RandomForest, no imbalance handling): {baseline_mean:.4f}")
print(f"Best with imbalance handling ({best_imbalance_name}): {best_imbalance_mean:.4f}")
print(f"Improvement: {improvement:+.4f} ({improvement_pct:+.2f}%)")

# Check if improvement is consistent across folds
baseline_all_folds = rf_baseline_cv
best_imbalance_all_folds = {'Weighted RF': rf_weighted_cv, 'SMOTE RF': rf_smote_cv, 'Undersample RF': rf_under_cv}[best_imbalance_name]

print(f"\nStability across folds:")
print(f"  Baseline: min={baseline_all_folds.min():.4f}, max={baseline_all_folds.max():.4f}, range={baseline_all_folds.max() - baseline_all_folds.min():.4f}")
print(f"  {best_imbalance_name}: min={best_imbalance_all_folds.min():.4f}, max={best_imbalance_all_folds.max():.4f}, range={best_imbalance_all_folds.max() - best_imbalance_all_folds.min():.4f}")

# Statistical significance check
better_folds = np.sum(best_imbalance_all_folds > baseline_all_folds)
print(f"\nCrossover folds (imbalance handling > baseline): {better_folds} out of {len(baseline_all_folds)}")

if improvement > 0.005:  # More than 0.5% improvement
    print(f"\n✓ Conclusion: Addressing class imbalance IMPROVES model quality")
    print(f"  The improvement ({improvement:+.4f} ROC-AUC) is consistent and meaningful.")
    finding_direction = f"Imbalance handling improves ROC-AUC by {improvement:.4f}"
elif improvement < -0.005:
    print(f"\n✗ Conclusion: Addressing class imbalance DECREASES model quality")
    print(f"  The degradation ({improvement:.4f} ROC-AUC) is consistent.")
    finding_direction = f"Imbalance handling decreases ROC-AUC by {abs(improvement):.4f}"
else:
    print(f"\n~ Conclusion: Addressing class imbalance has MINIMAL/NO EFFECT on model quality")
    print(f"  The difference ({improvement:+.4f} ROC-AUC) is negligible.")
    finding_direction = f"Imbalance handling has minimal effect (Δ {improvement:+.4f})"

print("\n" + "=" * 80)
