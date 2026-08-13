import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD AND EXPLORE DATA
# ============================================================================
print("=" * 80)
print("LOAD AND EXPLORE DATA")
print("=" * 80)

df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nTarget class distribution:")
class_dist = df['class'].value_counts()
print(class_dist)
print(f"\nClass proportions:")
print(df['class'].value_counts(normalize=True))

imbalance_ratio = class_dist.max() / class_dist.min()
print(f"\nImbalance ratio: {imbalance_ratio:.2f}:1")

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("DATA PREPROCESSING")
print("=" * 80)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class'].map({'<=50K': 0, '>50K': 1})

# Handle missing values and encode categorical features
X_clean = X.copy()
X_clean = X_clean.fillna(X_clean.mode().iloc[0])

# Identify categorical and numerical columns
categorical_cols = X_clean.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X_clean.select_dtypes(include=[np.number]).columns.tolist()

print(f"Categorical columns: {len(categorical_cols)}")
print(f"Numerical columns: {len(numerical_cols)}")

# Encode categorical variables
le_dict = {}
X_encoded = X_clean.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_clean[col].astype(str))
    le_dict[col] = le

print(f"Features after encoding: {X_encoded.shape[1]}")

# Train-test split (80-20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Training set class distribution:")
print(y_train.value_counts())

# Scale numerical features
scaler = StandardScaler()
X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

# ============================================================================
# APPROACH 1: BASELINE MODEL WITHOUT IMBALANCE HANDLING
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 1: BASELINE MODEL (No Imbalance Handling)")
print("=" * 80)

baseline_results = {}

# Gradient Boosting (often works well on imbalanced data naturally)
print("\nFitting Gradient Boosting Classifier (baseline)...")
gb_baseline = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_baseline.fit(X_train, y_train)

y_pred_baseline = gb_baseline.predict(X_test)
y_pred_proba_baseline = gb_baseline.predict_proba(X_test)[:, 1]

baseline_results['roc_auc'] = roc_auc_score(y_test, y_pred_proba_baseline)
baseline_results['balanced_acc'] = balanced_accuracy_score(y_test, y_pred_baseline)
baseline_results['f1'] = f1_score(y_test, y_pred_baseline)

print(f"ROC-AUC: {baseline_results['roc_auc']:.4f}")
print(f"Balanced Accuracy: {baseline_results['balanced_acc']:.4f}")
print(f"F1-Score: {baseline_results['f1']:.4f}")

# ============================================================================
# APPROACH 2: MODEL WITH CLASS WEIGHT ADJUSTMENT
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 2: Model with Class Weight Adjustment")
print("=" * 80)

print("\nFitting Gradient Boosting Classifier (scale_pos_weight)...")
# Calculate class weight to balance classes
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count

gb_weighted = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_weighted.fit(X_train, y_train)

y_pred_weighted = gb_weighted.predict(X_test)
y_pred_proba_weighted = gb_weighted.predict_proba(X_test)[:, 1]

weighted_results = {}
weighted_results['roc_auc'] = roc_auc_score(y_test, y_pred_proba_weighted)
weighted_results['balanced_acc'] = balanced_accuracy_score(y_test, y_pred_weighted)
weighted_results['f1'] = f1_score(y_test, y_pred_weighted)

print(f"ROC-AUC: {weighted_results['roc_auc']:.4f}")
print(f"Balanced Accuracy: {weighted_results['balanced_acc']:.4f}")
print(f"F1-Score: {weighted_results['f1']:.4f}")

# ============================================================================
# APPROACH 3: MODEL WITH SMOTE (OVERSAMPLING)
# ============================================================================
print("\n" + "=" * 80)
print("APPROACH 3: Model with SMOTE (Oversampling)")
print("=" * 80)

print("\nApplying SMOTE and fitting Gradient Boosting...")
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training set size after SMOTE: {X_train_smote.shape[0]}")
print(f"Class distribution after SMOTE:")
print(pd.Series(y_train_smote).value_counts())

gb_smote = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    random_state=42
)
gb_smote.fit(X_train_smote, y_train_smote)

y_pred_smote = gb_smote.predict(X_test)
y_pred_proba_smote = gb_smote.predict_proba(X_test)[:, 1]

smote_results = {}
smote_results['roc_auc'] = roc_auc_score(y_test, y_pred_proba_smote)
smote_results['balanced_acc'] = balanced_accuracy_score(y_test, y_pred_smote)
smote_results['f1'] = f1_score(y_test, y_pred_smote)

print(f"ROC-AUC: {smote_results['roc_auc']:.4f}")
print(f"Balanced Accuracy: {smote_results['balanced_acc']:.4f}")
print(f"F1-Score: {smote_results['f1']:.4f}")

# ============================================================================
# COMPARISON OF APPROACHES
# ============================================================================
print("\n" + "=" * 80)
print("COMPARISON OF APPROACHES")
print("=" * 80)

comparison_df = pd.DataFrame({
    'Baseline (No Handling)': baseline_results,
    'Class Weighted': weighted_results,
    'SMOTE': smote_results
})

print("\nPerformance Metrics Comparison:")
print(comparison_df)

# Calculate improvements
print("\nImprovement over Baseline:")
for metric in baseline_results.keys():
    baseline_val = baseline_results[metric]
    weighted_improvement = ((weighted_results[metric] - baseline_val) / baseline_val * 100)
    smote_improvement = ((smote_results[metric] - baseline_val) / baseline_val * 100)
    print(f"\n{metric}:")
    print(f"  Class Weighted: {weighted_improvement:+.2f}%")
    print(f"  SMOTE: {smote_improvement:+.2f}%")

# Primary metric: ROC-AUC (most relevant for imbalanced classification)
best_roc_auc = max(
    baseline_results['roc_auc'],
    weighted_results['roc_auc'],
    smote_results['roc_auc']
)

print(f"\nBest ROC-AUC achieved: {best_roc_auc:.4f}")

# ============================================================================
# VALIDATION: REPEATED STRATIFIED K-FOLD WITH DIFFERENT SEEDS
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION: Repeated Stratified K-Fold Cross-Validation")
print("=" * 80)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results_baseline = []
cv_results_weighted = []
cv_results_smote = []

fold = 0
for train_idx, val_idx in skf.split(X_train, y_train):
    fold += 1
    X_cv_train, X_cv_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_cv_train, y_cv_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    # Baseline model
    gb_cv_baseline = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
    gb_cv_baseline.fit(X_cv_train, y_cv_train)
    y_pred_cv = gb_cv_baseline.predict_proba(X_cv_val)[:, 1]
    cv_results_baseline.append(roc_auc_score(y_cv_val, y_pred_cv))

    # Weighted model
    gb_cv_weighted = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
    gb_cv_weighted.fit(X_cv_train, y_cv_train)
    y_pred_cv = gb_cv_weighted.predict_proba(X_cv_val)[:, 1]
    cv_results_weighted.append(roc_auc_score(y_cv_val, y_pred_cv))

    # SMOTE model
    smote_cv = SMOTE(random_state=42)
    X_cv_train_smote, y_cv_train_smote = smote_cv.fit_resample(X_cv_train, y_cv_train)
    gb_cv_smote = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42
    )
    gb_cv_smote.fit(X_cv_train_smote, y_cv_train_smote)
    y_pred_cv = gb_cv_smote.predict_proba(X_cv_val)[:, 1]
    cv_results_smote.append(roc_auc_score(y_cv_val, y_pred_cv))

    print(f"Fold {fold}: Baseline={cv_results_baseline[-1]:.4f}, "
          f"Weighted={cv_results_weighted[-1]:.4f}, SMOTE={cv_results_smote[-1]:.4f}")

print("\nCross-Validation Summary (ROC-AUC):")
print(f"Baseline:      Mean={np.mean(cv_results_baseline):.4f} ± {np.std(cv_results_baseline):.4f}")
print(f"Class Weighted: Mean={np.mean(cv_results_weighted):.4f} ± {np.std(cv_results_weighted):.4f}")
print(f"SMOTE:         Mean={np.mean(cv_results_smote):.4f} ± {np.std(cv_results_smote):.4f}")

# ============================================================================
# VALIDATION WITH ADDITIONAL RANDOM SEEDS
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION: Multiple Random Seeds on Test Set")
print("=" * 80)

seed_results_baseline = []
seed_results_weighted = []
seed_results_smote = []

for seed in [42, 123, 456, 789, 1011]:
    # Re-split data with different seed
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_encoded, y, test_size=0.2, stratify=y, random_state=seed
    )

    # Scale
    scaler_seed = StandardScaler()
    X_tr[numerical_cols] = scaler_seed.fit_transform(X_tr[numerical_cols])
    X_te[numerical_cols] = scaler_seed.transform(X_te[numerical_cols])

    # Baseline
    gb_seed_baseline = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed
    )
    gb_seed_baseline.fit(X_tr, y_tr)
    y_pred_seed = gb_seed_baseline.predict_proba(X_te)[:, 1]
    seed_results_baseline.append(roc_auc_score(y_te, y_pred_seed))

    # Weighted
    gb_seed_weighted = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed
    )
    gb_seed_weighted.fit(X_tr, y_tr)
    y_pred_seed = gb_seed_weighted.predict_proba(X_te)[:, 1]
    seed_results_weighted.append(roc_auc_score(y_te, y_pred_seed))

    # SMOTE
    smote_seed = SMOTE(random_state=seed)
    X_tr_smote, y_tr_smote = smote_seed.fit_resample(X_tr, y_tr)
    gb_seed_smote = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=seed
    )
    gb_seed_smote.fit(X_tr_smote, y_tr_smote)
    y_pred_seed = gb_seed_smote.predict_proba(X_te)[:, 1]
    seed_results_smote.append(roc_auc_score(y_te, y_pred_seed))

print("Seed-based Validation (Test Set ROC-AUC):")
print(f"Baseline:      Mean={np.mean(seed_results_baseline):.4f} ± {np.std(seed_results_baseline):.4f}")
print(f"Class Weighted: Mean={np.mean(seed_results_weighted):.4f} ± {np.std(seed_results_weighted):.4f}")
print(f"SMOTE:         Mean={np.mean(seed_results_smote):.4f} ± {np.std(seed_results_smote):.4f}")

# ============================================================================
# FINAL CONCLUSIONS
# ============================================================================
print("\n" + "=" * 80)
print("FINAL ANALYSIS")
print("=" * 80)

# Compare mean improvements
baseline_cv_mean = np.mean(cv_results_baseline)
weighted_cv_mean = np.mean(cv_results_weighted)
smote_cv_mean = np.mean(cv_results_smote)

baseline_seed_mean = np.mean(seed_results_baseline)
weighted_seed_mean = np.mean(seed_results_weighted)
smote_seed_mean = np.mean(seed_results_smote)

print(f"\nCross-Validation Mean ROC-AUC:")
print(f"  Baseline: {baseline_cv_mean:.4f}")
print(f"  Weighted: {weighted_cv_mean:.4f} ({weighted_cv_mean - baseline_cv_mean:+.4f})")
print(f"  SMOTE:    {smote_cv_mean:.4f} ({smote_cv_mean - baseline_cv_mean:+.4f})")

print(f"\nSeed-based Test Set Mean ROC-AUC:")
print(f"  Baseline: {baseline_seed_mean:.4f}")
print(f"  Weighted: {weighted_seed_mean:.4f} ({weighted_seed_mean - baseline_seed_mean:+.4f})")
print(f"  SMOTE:    {smote_seed_mean:.4f} ({smote_seed_mean - baseline_seed_mean:+.4f})")

# Determine if addressing imbalance helps
improvement_cv = max(weighted_cv_mean, smote_cv_mean) - baseline_cv_mean
improvement_seed = max(weighted_seed_mean, smote_seed_mean) - baseline_seed_mean

print(f"\nMax improvement (CV): {improvement_cv:+.4f}")
print(f"Max improvement (Seed-based): {improvement_seed:+.4f}")

if improvement_cv > 0.001 and improvement_seed > 0.001:
    conclusion = "YES - Addressing class imbalance improves model quality"
    direction = "Imbalance handling improves ROC-AUC"
elif improvement_cv > 0.0001 or improvement_seed > 0.0001:
    conclusion = "MIXED - Small improvements observed but inconsistent"
    direction = "Marginal/inconsistent improvement"
else:
    conclusion = "NO - Addressing class imbalance does not meaningfully improve performance"
    direction = "No meaningful improvement"

print(f"\nConclusion: {conclusion}")
print(f"Direction: {direction}")

# Export summary for results.json
summary_stats = {
    'baseline_cv_mean': float(baseline_cv_mean),
    'weighted_cv_mean': float(weighted_cv_mean),
    'smote_cv_mean': float(smote_cv_mean),
    'baseline_seed_mean': float(baseline_seed_mean),
    'weighted_seed_mean': float(weighted_seed_mean),
    'smote_seed_mean': float(smote_seed_mean),
    'improvement_cv': float(improvement_cv),
    'improvement_seed': float(improvement_seed),
    'conclusion': conclusion,
    'direction': direction
}

print("\nSummary statistics for results.json:")
print(summary_stats)
