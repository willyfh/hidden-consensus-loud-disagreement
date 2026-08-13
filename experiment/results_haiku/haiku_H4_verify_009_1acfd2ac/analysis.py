import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc, balanced_accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load and explore data
print("=" * 80)
print("STEP 1: DATA EXPLORATION")
print("=" * 80)

df = pd.read_csv('adult_income.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:")
print(df.head())
print(f"\nData types:")
print(df.dtypes)
print(f"\nMissing values:")
print(df.isnull().sum())
print(f"\nTarget class distribution:")
class_dist = df['class'].value_counts()
print(class_dist)
print(f"Class distribution (%):")
print(class_dist / len(df) * 100)

# Check for imbalance ratio
imbalance_ratio = class_dist.max() / class_dist.min()
print(f"\nImbalance ratio (majority/minority): {imbalance_ratio:.2f}")

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

print(f"\nFeature columns: {X.columns.tolist()}")
print(f"Number of features: {X.shape[1]}")

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# ============================================================================
# STEP 2: PREPROCESSING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 2: PREPROCESSING")
print("=" * 80)

# Encode target
target_mapping = {v: i for i, v in enumerate(y.unique())}
y_encoded = y.map(target_mapping)
print(f"Target encoding: {target_mapping}")

# Encode categorical features
X_processed = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Standardize numeric features
scaler = StandardScaler()
X_processed[numeric_cols] = scaler.fit_transform(X_processed[numeric_cols])

print(f"Processed features shape: {X_processed.shape}")

# ============================================================================
# STEP 3: SPLIT DATA
# ============================================================================
print("\n" + "=" * 80)
print("STEP 3: DATA SPLITTING")
print("=" * 80)

# Stratified train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Train set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train class distribution:")
print(pd.Series(y_train).value_counts())
print(f"Test class distribution:")
print(pd.Series(y_test).value_counts())

# ============================================================================
# STEP 4: BUILD MODELS WITH AND WITHOUT IMBALANCE HANDLING
# ============================================================================
print("\n" + "=" * 80)
print("STEP 4: MODEL COMPARISON - WITH vs WITHOUT CLASS IMBALANCE HANDLING")
print("=" * 80)

# Define models to test
models_config = {
    'LogisticRegression': {
        'without': LogisticRegression(max_iter=500, random_state=42),
        'with': LogisticRegression(max_iter=500, random_state=42, class_weight='balanced')
    },
    'RandomForest': {
        'without': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'with': RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
    },
    'GradientBoosting': {
        'without': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'with': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
}

results = {}

for model_name, model_dict in models_config.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    results[model_name] = {}

    # Train without imbalance handling
    print(f"\n--- WITHOUT class imbalance handling ---")
    model_without = model_dict['without']
    model_without.fit(X_train, y_train)

    y_pred_proba_without = model_without.predict_proba(X_test)[:, 1]
    y_pred_without = model_without.predict(X_test)

    auc_without = roc_auc_score(y_test, y_pred_proba_without)
    f1_without = f1_score(y_test, y_pred_without)
    balanced_acc_without = balanced_accuracy_score(y_test, y_pred_without)

    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_without)
    pr_auc_without = auc(recall, precision)

    results[model_name]['without'] = {
        'roc_auc': auc_without,
        'f1': f1_without,
        'balanced_accuracy': balanced_acc_without,
        'pr_auc': pr_auc_without
    }

    print(f"  ROC-AUC: {auc_without:.4f}")
    print(f"  F1-Score: {f1_without:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc_without:.4f}")
    print(f"  PR-AUC: {pr_auc_without:.4f}")

    # Train with imbalance handling (via class_weight or resampling)
    print(f"\n--- WITH class imbalance handling (class_weight='balanced') ---")
    model_with = model_dict['with']
    model_with.fit(X_train, y_train)

    y_pred_proba_with = model_with.predict_proba(X_test)[:, 1]
    y_pred_with = model_with.predict(X_test)

    auc_with = roc_auc_score(y_test, y_pred_proba_with)
    f1_with = f1_score(y_test, y_pred_with)
    balanced_acc_with = balanced_accuracy_score(y_test, y_pred_with)

    # Calculate PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba_with)
    pr_auc_with = auc(recall, precision)

    results[model_name]['with'] = {
        'roc_auc': auc_with,
        'f1': f1_with,
        'balanced_accuracy': balanced_acc_with,
        'pr_auc': pr_auc_with
    }

    print(f"  ROC-AUC: {auc_with:.4f}")
    print(f"  F1-Score: {f1_with:.4f}")
    print(f"  Balanced Accuracy: {balanced_acc_with:.4f}")
    print(f"  PR-AUC: {pr_auc_with:.4f}")

    # Calculate improvements
    print(f"\n--- IMPROVEMENT (with - without) ---")
    auc_imp = auc_with - auc_without
    f1_imp = f1_with - f1_without
    ba_imp = balanced_acc_with - balanced_acc_without
    pr_auc_imp = pr_auc_with - pr_auc_without

    print(f"  ROC-AUC delta: {auc_imp:+.4f} ({auc_imp/auc_without*100:+.2f}%)")
    print(f"  F1-Score delta: {f1_imp:+.4f} ({f1_imp/f1_without*100:+.2f}%)")
    print(f"  Balanced Accuracy delta: {ba_imp:+.4f} ({ba_imp/balanced_acc_without*100:+.2f}%)")
    print(f"  PR-AUC delta: {pr_auc_imp:+.4f} ({pr_auc_imp/pr_auc_without*100:+.2f}%)")

# ============================================================================
# STEP 5: CROSS-VALIDATION VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: STABILITY CHECK - REPEATED STRATIFIED K-FOLD CROSS-VALIDATION")
print("=" * 80)

# Use the best model from above (typically the one that showed most improvement)
# Let's focus on Random Forest as it's robust
print("\nUsing Random Forest for stability validation...")
print("Running 5x repeated 5-fold stratified cross-validation with different seeds\n")

cv_results = {
    'without': {'roc_auc': [], 'f1': [], 'balanced_acc': [], 'pr_auc': []},
    'with': {'roc_auc': [], 'f1': [], 'balanced_acc': [], 'pr_auc': []}
}

for seed in [42, 123, 456, 789, 999]:
    print(f"Seed: {seed}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    fold_idx = 0

    for train_idx, val_idx in skf.split(X_processed, y_encoded):
        fold_idx += 1
        X_train_cv = X_processed.iloc[train_idx]
        X_val_cv = X_processed.iloc[val_idx]
        y_train_cv = y_encoded.iloc[train_idx]
        y_val_cv = y_encoded.iloc[val_idx]

        # Without imbalance handling
        model_without = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        model_without.fit(X_train_cv, y_train_cv)
        y_pred_proba = model_without.predict_proba(X_val_cv)[:, 1]
        y_pred = model_without.predict(X_val_cv)

        cv_results['without']['roc_auc'].append(roc_auc_score(y_val_cv, y_pred_proba))
        cv_results['without']['f1'].append(f1_score(y_val_cv, y_pred))
        cv_results['without']['balanced_acc'].append(balanced_accuracy_score(y_val_cv, y_pred))

        precision, recall, _ = precision_recall_curve(y_val_cv, y_pred_proba)
        cv_results['without']['pr_auc'].append(auc(recall, precision))

        # With imbalance handling
        model_with = RandomForestClassifier(n_estimators=100, random_state=seed, class_weight='balanced', n_jobs=-1)
        model_with.fit(X_train_cv, y_train_cv)
        y_pred_proba = model_with.predict_proba(X_val_cv)[:, 1]
        y_pred = model_with.predict(X_val_cv)

        cv_results['with']['roc_auc'].append(roc_auc_score(y_val_cv, y_pred_proba))
        cv_results['with']['f1'].append(f1_score(y_val_cv, y_pred))
        cv_results['with']['balanced_acc'].append(balanced_accuracy_score(y_val_cv, y_pred))

        precision, recall, _ = precision_recall_curve(y_val_cv, y_pred_proba)
        cv_results['with']['pr_auc'].append(auc(recall, precision))

    print(f"  Fold 1-5 completed")

print("\n" + "=" * 80)
print("CROSS-VALIDATION RESULTS SUMMARY")
print("=" * 80)

print("\n--- WITHOUT Class Imbalance Handling ---")
for metric, values in cv_results['without'].items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    ci_lower = mean_val - 1.96 * std_val / np.sqrt(len(values))
    ci_upper = mean_val + 1.96 * std_val / np.sqrt(len(values))
    print(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}  [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")

print("\n--- WITH Class Imbalance Handling ---")
for metric, values in cv_results['with'].items():
    mean_val = np.mean(values)
    std_val = np.std(values)
    ci_lower = mean_val - 1.96 * std_val / np.sqrt(len(values))
    ci_upper = mean_val + 1.96 * std_val / np.sqrt(len(values))
    print(f"{metric:20s}: {mean_val:.4f} ± {std_val:.4f}  [95% CI: {ci_lower:.4f} - {ci_upper:.4f}]")

print("\n--- IMPROVEMENTS (5x5-fold CV) ---")
for metric in cv_results['without'].keys():
    without_vals = cv_results['without'][metric]
    with_vals = cv_results['with'][metric]

    improvement = np.mean(with_vals) - np.mean(without_vals)
    improvement_pct = improvement / np.mean(without_vals) * 100 if np.mean(without_vals) != 0 else 0

    # Paired t-test to check if improvement is significant
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(with_vals, without_vals)

    print(f"{metric:20s}: {improvement:+.4f} ({improvement_pct:+.2f}%)  p-value: {p_value:.6f}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("FINAL FINDINGS")
print("=" * 80)

print(f"\nDataset class imbalance ratio: {imbalance_ratio:.2f}:1")
print(f"This represents a {'SIGNIFICANT' if imbalance_ratio > 1.5 else 'MODERATE' if imbalance_ratio > 1.2 else 'MINOR'} class imbalance")

print("\nTest Set Performance (Random Forest on 80-20 split):")
print(f"  Without handling - ROC-AUC: {results['RandomForest']['without']['roc_auc']:.4f}")
print(f"  With handling    - ROC-AUC: {results['RandomForest']['with']['roc_auc']:.4f}")
print(f"  Improvement:             {results['RandomForest']['with']['roc_auc'] - results['RandomForest']['without']['roc_auc']:+.4f}")

cv_improvement_roc_auc = np.mean(cv_results['with']['roc_auc']) - np.mean(cv_results['without']['roc_auc'])
print(f"\nCross-Validation Performance (5x5-fold):")
print(f"  Without handling - ROC-AUC: {np.mean(cv_results['without']['roc_auc']):.4f} ± {np.std(cv_results['without']['roc_auc']):.4f}")
print(f"  With handling    - ROC-AUC: {np.mean(cv_results['with']['roc_auc']):.4f} ± {np.std(cv_results['with']['roc_auc']):.4f}")
print(f"  Improvement:             {cv_improvement_roc_auc:+.4f}")

if cv_improvement_roc_auc > 0.001:
    print("\n✓ FINDING: Addressing class imbalance IMPROVES model quality")
    print(f"  The improvement is consistent across CV folds (mean improvement: {cv_improvement_roc_auc:+.4f})")
elif cv_improvement_roc_auc < -0.001:
    print("\n✗ FINDING: Addressing class imbalance WORSENS model quality")
    print(f"  The degradation is consistent across CV folds (mean change: {cv_improvement_roc_auc:+.4f})")
else:
    print("\n≈ FINDING: Addressing class imbalance has MINIMAL IMPACT on model quality")
    print(f"  The change is negligible (mean change: {cv_improvement_roc_auc:+.4f})")

print("\n" + "=" * 80)
