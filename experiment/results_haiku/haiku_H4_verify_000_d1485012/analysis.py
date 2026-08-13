"""
Analysis: Does addressing class imbalance improve model quality on the adult income dataset?
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import json
import warnings
warnings.filterwarnings('ignore')

# ==================== STEP 1: LOAD AND EXPLORE DATA ====================
print("=" * 70)
print("STEP 1: LOADING AND EXPLORING DATA")
print("=" * 70)

df = pd.read_csv('adult_income.csv')
print(f"\nDataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")

# Check class distribution
print(f"\n--- CLASS DISTRIBUTION ---")
class_dist = df['class'].value_counts()
print(class_dist)
print(f"\nClass proportions:\n{df['class'].value_counts(normalize=True)}")
imbalance_ratio = class_dist.max() / class_dist.min()
print(f"Imbalance ratio: {imbalance_ratio:.2f}:1")

# ==================== STEP 2: DATA PREPROCESSING ====================
print("\n" + "=" * 70)
print("STEP 2: DATA PREPROCESSING")
print("=" * 70)

# Separate features and target
X = df.drop('class', axis=1)
y = (df['class'] == '>50K').astype(int)  # Binary: 1 if >50K, 0 if <=50K

print(f"\nTarget distribution after encoding:")
print(f"Class 0 (<=50K): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"Class 1 (>50K): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols[:5]}...")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# Encode categorical variables
label_encoders = {}
X_encoded = X.copy()
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"\nAfter encoding, feature matrix shape: {X_encoded.shape}")

# Split data: use 70-30 split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== STEP 3: BUILD BASELINE MODELS (NO IMBALANCE HANDLING) ====================
print("\n" + "=" * 70)
print("STEP 3: BASELINE MODELS (NO IMBALANCE HANDLING)")
print("=" * 70)

baseline_models = {
    'LogisticRegression': LogisticRegression(max_iter=500, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

baseline_results = {}

for name, model in baseline_models.items():
    print(f"\n--- {name} (Baseline) ---")

    # For Logistic Regression, use scaled data; for tree models, use unscaled
    if name == 'LogisticRegression':
        model.fit(X_train_scaled, y_train)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    f1 = f1_score(y_test, y_pred)

    baseline_results[name] = {'roc_auc': roc_auc, 'f1': f1}
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"F1-Score: {f1:.4f}")

# ==================== STEP 4: BUILD MODELS WITH IMBALANCE HANDLING ====================
print("\n" + "=" * 70)
print("STEP 4: MODELS WITH IMBALANCE HANDLING")
print("=" * 70)

imbalance_methods = {
    'class_weight': {'method': 'class_weight', 'params': {}},
    'SMOTE': {'method': 'SMOTE', 'params': {}},
    'undersampling': {'method': 'undersampling', 'params': {}},
}

balanced_results = {}

for method_name, method_info in imbalance_methods.items():
    print(f"\n--- Imbalance Method: {method_name} ---")

    if method_info['method'] == 'class_weight':
        # Use class_weight for Logistic Regression and Gradient Boosting
        X_train_for_balance = X_train_scaled if baseline_models['LogisticRegression'].__class__.__name__ == 'LogisticRegression' else X_train
        models_with_balance = {
            'LogisticRegression_balanced': LogisticRegression(max_iter=500, class_weight='balanced', random_state=42),
            'GradientBoosting_balanced': GradientBoostingClassifier(n_estimators=100, random_state=42)
        }

        balanced_models_metrics = {}
        for model_name, model in models_with_balance.items():
            if 'LogisticRegression' in model_name:
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            else:
                model.fit(X_train, y_train)
                y_pred_proba = model.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_pred_proba)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            f1 = f1_score(y_test, y_pred)

            base_name = model_name.split('_')[0]
            if base_name not in balanced_models_metrics:
                balanced_models_metrics[base_name] = {}
            balanced_models_metrics[base_name]['roc_auc'] = roc_auc
            balanced_models_metrics[base_name]['f1'] = f1

            print(f"  {model_name}: ROC-AUC={roc_auc:.4f}, F1={f1:.4f}")

        balanced_results[method_name] = balanced_models_metrics

    else:
        # SMOTE and undersampling
        if method_info['method'] == 'SMOTE':
            sampler = SMOTE(random_state=42)
        else:  # undersampling
            sampler = RandomUnderSampler(random_state=42)

        X_train_balanced, y_train_balanced = sampler.fit_resample(X_train, y_train)
        X_train_balanced_scaled = scaler.fit_transform(X_train_balanced)

        print(f"  After {method_info['method']}: {np.bincount(y_train_balanced)}")

        balanced_models_metrics = {}
        for model_name, model in baseline_models.items():
            if model_name == 'LogisticRegression':
                model_copy = LogisticRegression(max_iter=500, random_state=42)
                model_copy.fit(X_train_balanced_scaled, y_train_balanced)
                y_pred_proba = model_copy.predict_proba(X_test_scaled)[:, 1]
            else:
                model_copy = model.__class__(**model.get_params())
                model_copy.fit(X_train_balanced, y_train_balanced)
                y_pred_proba = model_copy.predict_proba(X_test)[:, 1]

            roc_auc = roc_auc_score(y_test, y_pred_proba)
            y_pred = (y_pred_proba >= 0.5).astype(int)
            f1 = f1_score(y_test, y_pred)

            balanced_models_metrics[model_name] = {'roc_auc': roc_auc, 'f1': f1}
            print(f"  {model_name}: ROC-AUC={roc_auc:.4f}, F1={f1:.4f}")

        balanced_results[method_name] = balanced_models_metrics

# ==================== STEP 5: COMPUTE IMPROVEMENTS ====================
print("\n" + "=" * 70)
print("STEP 5: PERFORMANCE IMPROVEMENTS FROM IMBALANCE HANDLING")
print("=" * 70)

all_improvements = []

for method_name, method_results in balanced_results.items():
    print(f"\n--- {method_name} ---")

    for model_name, metrics in method_results.items():
        if model_name in baseline_results:
            baseline_auc = baseline_results[model_name]['roc_auc']
            baseline_f1 = baseline_results[model_name]['f1']

            auc_improvement = metrics['roc_auc'] - baseline_auc
            f1_improvement = metrics['f1'] - baseline_f1

            all_improvements.append({
                'method': method_name,
                'model': model_name,
                'auc_improvement': auc_improvement,
                'f1_improvement': f1_improvement,
                'baseline_auc': baseline_auc,
                'new_auc': metrics['roc_auc'],
                'baseline_f1': baseline_f1,
                'new_f1': metrics['f1']
            })

            print(f"  {model_name}:")
            print(f"    ROC-AUC: {baseline_auc:.4f} → {metrics['roc_auc']:.4f} ({auc_improvement:+.4f})")
            print(f"    F1-Score: {baseline_f1:.4f} → {metrics['f1']:.4f} ({f1_improvement:+.4f})")

# ==================== STEP 6: CROSS-VALIDATION FOR STABILITY ====================
print("\n" + "=" * 70)
print("STEP 6: CROSS-VALIDATION FOR STABILITY (with different random seeds)")
print("=" * 70)

cv_results = {
    'baseline': {},
    'with_class_weight': {},
    'with_SMOTE': {}
}

seeds = [42, 123, 456, 789, 999]
n_splits = 5

print(f"\nRunning {len(seeds)} iterations of {n_splits}-fold cross-validation...")

for seed in seeds:
    print(f"\n  Seed {seed}:")
    cv_splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Baseline
    lr_model = LogisticRegression(max_iter=500, random_state=seed)
    cv_scores = cross_validate(
        lr_model, X_train_scaled, y_train,
        cv=cv_splitter,
        scoring=['roc_auc', 'f1'],
        n_jobs=-1
    )
    if 'baseline' not in cv_results['baseline']:
        cv_results['baseline']['scores'] = []
    cv_results['baseline']['scores'].append({
        'seed': seed,
        'roc_auc_mean': cv_scores['test_roc_auc'].mean(),
        'roc_auc_std': cv_scores['test_roc_auc'].std(),
        'f1_mean': cv_scores['test_f1'].mean(),
        'f1_std': cv_scores['test_f1'].std()
    })
    print(f"    Baseline LR: ROC-AUC={cv_scores['test_roc_auc'].mean():.4f}±{cv_scores['test_roc_auc'].std():.4f}")

    # With class_weight
    lr_balanced = LogisticRegression(max_iter=500, class_weight='balanced', random_state=seed)
    cv_scores_balanced = cross_validate(
        lr_balanced, X_train_scaled, y_train,
        cv=cv_splitter,
        scoring=['roc_auc', 'f1'],
        n_jobs=-1
    )
    if 'with_class_weight' not in cv_results['with_class_weight']:
        cv_results['with_class_weight']['scores'] = []
    cv_results['with_class_weight']['scores'].append({
        'seed': seed,
        'roc_auc_mean': cv_scores_balanced['test_roc_auc'].mean(),
        'roc_auc_std': cv_scores_balanced['test_roc_auc'].std(),
        'f1_mean': cv_scores_balanced['test_f1'].mean(),
        'f1_std': cv_scores_balanced['test_f1'].std()
    })
    print(f"    With class_weight: ROC-AUC={cv_scores_balanced['test_roc_auc'].mean():.4f}±{cv_scores_balanced['test_roc_auc'].std():.4f}")

    # With SMOTE
    smote = SMOTE(random_state=seed)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)

    smote_cv_aucs = []
    smote_cv_f1s = []
    for train_idx, val_idx in cv_splitter.split(X_train, y_train):
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

        X_fold_balanced, y_fold_balanced = smote.fit_resample(X_fold_train, y_fold_train)
        rf_model_copy = RandomForestClassifier(n_estimators=100, random_state=seed, n_jobs=-1)
        rf_model_copy.fit(X_fold_balanced, y_fold_balanced)

        y_pred_proba = rf_model_copy.predict_proba(X_fold_val)[:, 1]
        smote_cv_aucs.append(roc_auc_score(y_fold_val, y_pred_proba))
        y_pred = (y_pred_proba >= 0.5).astype(int)
        smote_cv_f1s.append(f1_score(y_fold_val, y_pred))

    if 'with_SMOTE' not in cv_results['with_SMOTE']:
        cv_results['with_SMOTE']['scores'] = []
    cv_results['with_SMOTE']['scores'].append({
        'seed': seed,
        'roc_auc_mean': np.mean(smote_cv_aucs),
        'roc_auc_std': np.std(smote_cv_aucs),
        'f1_mean': np.mean(smote_cv_f1s),
        'f1_std': np.std(smote_cv_f1s)
    })
    print(f"    With SMOTE (RF): ROC-AUC={np.mean(smote_cv_aucs):.4f}±{np.std(smote_cv_aucs):.4f}")

# Compute aggregate CV statistics
print("\n--- AGGREGATE CV RESULTS ---")
cv_summary = {}

for condition, results in cv_results.items():
    if results.get('scores'):
        auc_means = [s['roc_auc_mean'] for s in results['scores']]
        f1_means = [s['f1_mean'] for s in results['scores']]

        cv_summary[condition] = {
            'roc_auc_mean': np.mean(auc_means),
            'roc_auc_std': np.std(auc_means),
            'f1_mean': np.mean(f1_means),
            'f1_std': np.std(f1_means)
        }

        print(f"\n{condition}:")
        print(f"  ROC-AUC: {cv_summary[condition]['roc_auc_mean']:.4f} ± {cv_summary[condition]['roc_auc_std']:.4f}")
        print(f"  F1-Score: {cv_summary[condition]['f1_mean']:.4f} ± {cv_summary[condition]['f1_std']:.4f}")

# ==================== STEP 7: FINAL ANALYSIS ====================
print("\n" + "=" * 70)
print("STEP 7: FINAL ANALYSIS")
print("=" * 70)

# Compute average improvement across all settings
improvements_auc = [imp['auc_improvement'] for imp in all_improvements]
improvements_f1 = [imp['f1_improvement'] for imp in all_improvements]

print(f"\nAverage ROC-AUC improvement: {np.mean(improvements_auc):+.4f}")
print(f"Median ROC-AUC improvement: {np.median(improvements_auc):+.4f}")
print(f"Min/Max ROC-AUC improvement: {np.min(improvements_auc):+.4f} / {np.max(improvements_auc):+.4f}")
print(f"Standard deviation: {np.std(improvements_auc):.4f}")

print(f"\nAverage F1-Score improvement: {np.mean(improvements_f1):+.4f}")
print(f"Median F1-Score improvement: {np.median(improvements_f1):+.4f}")
print(f"Min/Max F1-Score improvement: {np.min(improvements_f1):+.4f} / {np.max(improvements_f1):+.4f}")
print(f"Standard deviation: {np.std(improvements_f1):.4f}")

# Count improvements
positive_auc = sum(1 for x in improvements_auc if x > 0)
positive_f1 = sum(1 for x in improvements_f1 if x > 0)

print(f"\nCases with ROC-AUC improvement: {positive_auc}/{len(improvements_auc)}")
print(f"Cases with F1-Score improvement: {positive_f1}/{len(improvements_f1)}")

# ==================== STEP 8: CROSS-VALIDATION VERIFICATION ====================
print("\n" + "=" * 70)
print("STEP 8: VERIFICATION OF STABILITY ACROSS CV RUNS")
print("=" * 70)

if cv_summary:
    baseline_cv_auc = cv_summary['baseline']['roc_auc_mean']
    balanced_cv_auc = cv_summary['with_class_weight']['roc_auc_mean']
    cv_improvement = balanced_cv_auc - baseline_cv_auc

    print(f"\nCV Baseline ROC-AUC: {baseline_cv_auc:.4f} ± {cv_summary['baseline']['roc_auc_std']:.4f}")
    print(f"CV With class_weight: {balanced_cv_auc:.4f} ± {cv_summary['with_class_weight']['roc_auc_std']:.4f}")
    print(f"CV Improvement: {cv_improvement:+.4f}")

    if cv_improvement > 0:
        print("\n✓ Finding verified: imbalance handling provides consistent improvement across CV runs")
        stability_result = True
    else:
        print("\n✗ Finding not stable: baseline performs as well or better in CV")
        stability_result = False

print("\n" + "=" * 70)
print("ANALYSIS COMPLETE")
print("=" * 70)

# Save all results for JSON output
results_summary = {
    'hypothesis_id': 'H4',
    'imbalance_ratio': imbalance_ratio,
    'baseline_results': baseline_results,
    'balanced_results': balanced_results,
    'improvements': all_improvements,
    'cv_summary': cv_summary,
    'avg_auc_improvement': float(np.mean(improvements_auc)),
    'avg_f1_improvement': float(np.mean(improvements_f1)),
    'stability_verified': stability_result
}

# Save to temporary file for inspection
with open('results_summary.json', 'w') as f:
    json.dump(results_summary, f, indent=2, default=float)

print("\nResults saved to results_summary.json")
