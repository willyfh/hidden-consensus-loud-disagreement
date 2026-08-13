"""
Analysis: Does the choice of model family meaningfully affect predictive performance?

Hypothesis H1: We test whether different model families (e.g., linear, tree-based,
ensemble methods) produce substantially different predictive performance on the adult
income dataset.

Methodology:
- Preprocessing: Handle missing values, encode categorical features, scale numeric features
- Models compared: Logistic Regression, Random Forest, Gradient Boosting, AdaBoost, SVM
- Evaluation: ROC-AUC score (robust to class imbalance)
- Validation: 3-fold CV with 5 different random seeds to assess stability
- Statistical test: Calculate confidence intervals across runs to determine if differences
  between top and bottom model families are meaningful
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
import json
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: LOAD AND PREPROCESS DATA
# ============================================================================

df = pd.read_csv('adult_income.csv')

# Separate features and target
X = df.drop('class', axis=1)
y = df['class'].map({'<=50K': 0, '>50K': 1})

# Identify column types
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"Numeric columns: {numeric_cols}")
print(f"Categorical columns: {categorical_cols}")

# Handle missing values
# For numeric: fill with median
X_numeric = X[numeric_cols].fillna(X[numeric_cols].median())

# For categorical: fill with most frequent value
X_categorical = X[categorical_cols].fillna(X[categorical_cols].mode().iloc[0])

# Combine
X_clean = pd.concat([X_numeric, X_categorical], axis=1)

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_clean[col] = le.fit_transform(X_clean[col].astype(str))
    label_encoders[col] = le

# ============================================================================
# STEP 2: DEFINE MODEL FAMILIES
# ============================================================================

models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000,
        random_state=42,
        solver='lbfgs',
        n_jobs=-1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        n_jobs=-1,
        max_depth=20
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=50,
        random_state=42,
        learning_rate=0.1,
        max_depth=5
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=50,
        random_state=42
    ),
    'SVM': SVC(
        kernel='rbf',
        probability=True,
        random_state=42,
        max_iter=1000
    ),
}

# ============================================================================
# STEP 3: EVALUATE MODELS WITH CROSS-VALIDATION
# ============================================================================

# Create a preprocessing pipeline (scale numeric features only)
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', 'passthrough', categorical_cols)
    ]
)

print("\n" + "="*70)
print("INITIAL EVALUATION: 3-Fold Cross-Validation")
print("="*70)

initial_results = {}

for model_name, model in models.items():
    # Create pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Evaluate with 3-fold CV
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_clean, y, cv=cv, scoring='roc_auc', n_jobs=1)

    initial_results[model_name] = {
        'scores': scores.tolist(),
        'mean': float(scores.mean()),
        'std': float(scores.std())
    }

    print(f"\n{model_name:20s}: {scores.mean():.4f} ± {scores.std():.4f}")
    print(f"  Fold scores: {[f'{s:.4f}' for s in scores]}")

# ============================================================================
# STEP 4: STABILITY VALIDATION WITH REPEATED CV (DIFFERENT SEEDS)
# ============================================================================

print("\n" + "="*70)
print("STABILITY CHECK: 3-Fold CV with 5 Different Random Seeds")
print("="*70)

validation_results = {}
all_seeds_results = {}

for model_name, model in models.items():
    print(f"\nProcessing {model_name}...", end=" ")
    all_fold_scores = []
    seed_means = []

    for seed in [42, 123, 456, 789, 999]:
        pipeline = Pipeline([
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])

        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        scores = cross_val_score(pipeline, X_clean, y, cv=cv, scoring='roc_auc', n_jobs=1)

        all_fold_scores.extend(scores)
        seed_means.append(scores.mean())

    all_seeds_results[model_name] = {
        'all_fold_scores': all_fold_scores,
        'seed_means': seed_means
    }

    mean_across_seeds = np.mean(seed_means)
    std_across_seeds = np.std(seed_means)
    ci_95 = 1.96 * std_across_seeds

    validation_results[model_name] = {
        'mean_roc_auc': float(mean_across_seeds),
        'std_roc_auc': float(std_across_seeds),
        'ci_95_margin': float(ci_95),
        'lower_ci': float(mean_across_seeds - ci_95),
        'upper_ci': float(mean_across_seeds + ci_95),
        'seed_wise_means': [float(m) for m in seed_means]
    }

    print(f"✓")

print("\n--- Results by Model ---")
for model_name, results in validation_results.items():
    print(f"\n{model_name:20s}:")
    print(f"  Mean ROC-AUC across seeds: {results['mean_roc_auc']:.4f} ± {results['std_roc_auc']:.4f}")
    print(f"  95% CI: [{results['lower_ci']:.4f}, {results['upper_ci']:.4f}]")
    print(f"  Seed-wise means: {[f'{m:.4f}' for m in results['seed_wise_means']]}")

# ============================================================================
# STEP 5: STATISTICAL COMPARISON
# ============================================================================

print("\n" + "="*70)
print("STATISTICAL COMPARISON")
print("="*70)

sorted_models = sorted(
    validation_results.items(),
    key=lambda x: x[1]['mean_roc_auc'],
    reverse=True
)

print("\nModels ranked by ROC-AUC (with 95% CI):")
for rank, (model_name, results) in enumerate(sorted_models, 1):
    print(f"{rank}. {model_name:20s}: {results['mean_roc_auc']:.4f} "
          f"[{results['lower_ci']:.4f}, {results['upper_ci']:.4f}]")

best_model = sorted_models[0][0]
worst_model = sorted_models[-1][0]
best_performance = sorted_models[0][1]['mean_roc_auc']
worst_performance = sorted_models[-1][1]['mean_roc_auc']
performance_gap = best_performance - worst_performance

print(f"\n--- Performance Gap Analysis ---")
print(f"Best model:  {best_model} ({best_performance:.4f})")
print(f"Worst model: {worst_model} ({worst_performance:.4f})")
print(f"Absolute difference: {performance_gap:.4f}")
print(f"Relative difference: {(performance_gap / worst_performance * 100):.2f}%")

# Check if CIs overlap
best_ci = sorted_models[0][1]
worst_ci = sorted_models[-1][1]
ci_overlap = best_ci['lower_ci'] < worst_ci['upper_ci']

print(f"\n--- Confidence Interval Overlap ---")
print(f"Best model CI:  [{best_ci['lower_ci']:.4f}, {best_ci['upper_ci']:.4f}]")
print(f"Worst model CI: [{worst_ci['lower_ci']:.4f}, {worst_ci['upper_ci']:.4f}]")
print(f"CIs overlap: {ci_overlap}")

# ============================================================================
# STEP 6: DETERMINE IF DIFFERENCE IS MEANINGFUL
# ============================================================================

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

# Consider >1% relative difference as meaningful
threshold_relative = 0.01
is_meaningful = performance_gap > threshold_relative * worst_performance

print(f"\nThreshold for 'meaningful' difference: >{threshold_relative*100}% relative")
print(f"Observed relative difference: {performance_gap / worst_performance * 100:.2f}%")
print(f"Is difference meaningful? {is_meaningful}")

if not ci_overlap:
    print("\nCI does NOT overlap → Difference is statistically significant")
    finding = "Yes - model family choice meaningfully affects performance"
else:
    print("\nCI overlaps → Difference may not be statistically significant")
    if performance_gap > 0.01:
        finding = "Possible - model family may affect performance, but with uncertainty"
    else:
        finding = "No - model family choice does not meaningfully affect performance"

print(f"\nConclusion: {finding}")

# ============================================================================
# STEP 7: SAVE RESULTS
# ============================================================================

result = {
    "hypothesis_id": "H1",
    "summary": f"Model family choice {'meaningfully affects' if is_meaningful and not ci_overlap else 'does not substantially affect'} "
              f"predictive performance on the adult income dataset. "
              f"The best performing model ({best_model}, ROC-AUC={best_performance:.4f}) "
              f"outperforms the worst ({worst_model}, ROC-AUC={worst_performance:.4f}) "
              f"by {performance_gap:.4f} ({performance_gap / worst_performance * 100:.1f}%), "
              f"with {'non-overlapping' if not ci_overlap else 'overlapping'} confidence intervals across repeated validation runs.",
    "primary_metric_name": "ROC-AUC difference (best - worst model)",
    "primary_metric_value": float(performance_gap),
    "direction": f"{best_model} > {worst_model}",
    "methodological_choices": (
        "Preprocessing: Missing values filled with median (numeric) or mode (categorical); "
        "categorical variables label-encoded; numeric features standardized via StandardScaler. "
        "Models compared: Logistic Regression (linear), Random Forest (tree-based ensemble, 50 trees, max_depth=20), "
        "Gradient Boosting (gradient-based ensemble, 50 estimators, max_depth=5), "
        "AdaBoost (adaptive boosting ensemble, 50 estimators), and SVM with RBF kernel (kernel method). "
        "Evaluation metric: ROC-AUC (robust to class imbalance, ~76% negative / ~24% positive class split). "
        "Validation: 3-fold stratified cross-validation repeated with 5 different random seeds (42, 123, 456, 789, 999) "
        "to assess stability across different data splits. Confidence intervals calculated as mean ± 1.96*std across seed-wise means."
    ),
    "verification_method": "3x repeated 3-fold stratified cross-validation with different random seeds (total 15 fold evaluations per model). "
                          "Computed 95% confidence intervals around mean ROC-AUC for each model across all seed runs.",
    "verification_result": (
        f"Finding {'held up' if performance_gap > 0.01 else 'did not hold up'} under repeated validation. "
        f"Top 3 models: {', '.join([f'{m[0]} ({m[1]['mean_roc_auc']:.4f})' for m in sorted_models[:3]])}. "
        f"Performance differences were {'consistent' if std_across_seeds < 0.01 else 'variable'} across seeds. "
        f"Confidence intervals {'did not overlap between best and worst' if not ci_overlap else 'overlapped'}, "
        f"suggesting {'statistical significance' if not ci_overlap else 'no strong statistical difference'}."
    ),
    "detailed_validation_results": validation_results
}

with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*70)
print("Results saved to result.json")
print("="*70)
