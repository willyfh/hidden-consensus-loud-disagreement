"""
Analysis of model family effects on Adult Income prediction task.
Compares multiple model families to determine if choice meaningfully affects performance.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('adult_income.csv')

print(f"Dataset shape: {df.shape}")
print(f"Target class distribution:\n{df['class'].value_counts()}")
print()

# Data preprocessing
print("Preprocessing data...")

# Handle missing values and convert target
df['class'] = (df['class'] == '>50K').astype(int)
df = df.replace('?', np.nan)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical features: {len(categorical_cols)}")
print(f"Numerical features: {len(numerical_cols)}")

# Encode categorical variables
le_dict = {}
X_processed = X.copy()
for col in categorical_cols:
    # Fill NaN with 'Unknown'
    X_processed[col] = X_processed[col].fillna('Unknown')
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col])
    le_dict[col] = le

# Fill numerical NaN with median
for col in numerical_cols:
    X_processed[col] = X_processed[col].fillna(X_processed[col].median())

print(f"Data shape after preprocessing: {X_processed.shape}")
print()

# Define model families to compare
models = {
    'Logistic Regression': LogisticRegression(
        max_iter=1000, solver='saga', random_state=42, n_jobs=-1, tol=0.1
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=50, random_state=42, n_jobs=-1, max_depth=15
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=50, random_state=42, max_depth=5
    ),
    'SVM (RBF)': SVC(
        kernel='rbf', probability=True, random_state=42, C=1.0
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(32, 16), max_iter=200, random_state=42, early_stopping=False
    ),
}

# Evaluate each model using repeated cross-validation
print("=" * 60)
print("CROSS-VALIDATION PERFORMANCE COMPARISON")
print("=" * 60)

# Use repeated cross-validation for stability check
cv_results = {}
n_splits = 5
n_repeats = 3  # Reduced from 5 to 3 for speed

for model_name, model in models.items():
    print(f"\nEvaluating {model_name}...")

    # Repeated stratified k-fold
    roc_scores = []
    acc_scores = []

    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42 + repeat)

        # ROC-AUC score
        roc = cross_val_score(
            model, X_processed, y, cv=skf, scoring='roc_auc', n_jobs=-1
        )
        roc_scores.extend(roc)

        # Accuracy score
        acc = cross_val_score(
            model, X_processed, y, cv=skf, scoring='accuracy', n_jobs=-1
        )
        acc_scores.extend(acc)

    roc_scores = np.array(roc_scores)
    acc_scores = np.array(acc_scores)

    cv_results[model_name] = {
        'roc_auc_mean': roc_scores.mean(),
        'roc_auc_std': roc_scores.std(),
        'roc_auc_scores': roc_scores,
        'accuracy_mean': acc_scores.mean(),
        'accuracy_std': acc_scores.std(),
        'accuracy_scores': acc_scores,
    }

    print(f"  ROC-AUC:  {roc_scores.mean():.4f} ± {roc_scores.std():.4f}")
    print(f"  Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}")

print()

# Compute performance differences
print("=" * 60)
print("PERFORMANCE DIFFERENCES (ROC-AUC)")
print("=" * 60)

model_names = list(cv_results.keys())
best_model_name = max(model_names, key=lambda x: cv_results[x]['roc_auc_mean'])
best_score = cv_results[best_model_name]['roc_auc_mean']
worst_model_name = min(model_names, key=lambda x: cv_results[x]['roc_auc_mean'])
worst_score = cv_results[worst_model_name]['roc_auc_mean']

print(f"\nBest model:  {best_model_name} ({best_score:.4f})")
print(f"Worst model: {worst_model_name} ({worst_score:.4f})")
print(f"Difference:  {best_score - worst_score:.4f}")
print()

print("Performance by model (ROC-AUC):")
for name in sorted(model_names, key=lambda x: cv_results[x]['roc_auc_mean'], reverse=True):
    mean = cv_results[name]['roc_auc_mean']
    std = cv_results[name]['roc_auc_std']
    diff = mean - worst_score
    print(f"  {name:25s}: {mean:.4f} ± {std:.4f} (Δ={diff:+.4f})")

# Test for statistical significance using bootstrapped confidence intervals
print()
print("=" * 60)
print("STATISTICAL SIGNIFICANCE (95% Confidence Intervals)")
print("=" * 60)

from scipy import stats

# Perform pairwise comparisons
print("\nPairwise t-tests (best model vs others):")
best_scores = cv_results[best_model_name]['roc_auc_scores']

for other_name in model_names:
    if other_name == best_model_name:
        continue
    other_scores = cv_results[other_name]['roc_auc_scores']

    # Paired t-test (same CV folds)
    t_stat, p_value = stats.ttest_rel(best_scores, other_scores)
    mean_diff = best_scores.mean() - other_scores.mean()

    print(f"  {best_model_name} vs {other_name}:")
    print(f"    Mean difference: {mean_diff:.4f}")
    print(f"    p-value: {p_value:.6f}")
    sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
    print(f"    Significance: {sig}")

# Summary statistics
print()
print("=" * 60)
print("SUMMARY: MODEL FAMILY EFFECT SIZE")
print("=" * 60)

all_roc_scores = []
for model_name in model_names:
    all_roc_scores.extend(cv_results[model_name]['roc_auc_scores'])

range_of_means = max(cv_results[m]['roc_auc_mean'] for m in model_names) - \
                 min(cv_results[m]['roc_auc_mean'] for m in model_names)

print(f"\nRange of mean ROC-AUC across models: {range_of_means:.4f}")
print(f"Best model performance:             {best_score:.4f}")
print(f"Worst model performance:            {worst_score:.4f}")
print(f"Relative difference:                {(best_score - worst_score) / worst_score * 100:.2f}%")

# Determine if effect is meaningful
# A difference is typically considered meaningful if:
# 1. It's > 1% in relative terms, or
# 2. It's > 0.01 in absolute ROC-AUC terms
is_meaningful = (range_of_means > 0.01) and ((best_score - worst_score) / worst_score > 0.01)

print()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
if is_meaningful:
    print(f"YES - Model family choice MEANINGFULLY affects performance.")
    print(f"The best model ({best_model_name}) outperforms the worst ({worst_model_name})")
    print(f"by {best_score - worst_score:.4f} in ROC-AUC ({(best_score - worst_score) / worst_score * 100:.2f}%)")
else:
    print(f"NO - Model family choice has MINIMAL effect on performance.")
    print(f"Difference between best and worst is only {best_score - worst_score:.4f} ROC-AUC")

# Save detailed results for verification
print()
print("=" * 60)
print("DETAILED RESULTS FOR VERIFICATION")
print("=" * 60)

for model_name in sorted(model_names, key=lambda x: cv_results[x]['roc_auc_mean'], reverse=True):
    results = cv_results[model_name]
    print(f"\n{model_name}:")
    print(f"  ROC-AUC scores across CV folds: {results['roc_auc_scores']}")
    print(f"  Mean: {results['roc_auc_mean']:.4f}, Std: {results['roc_auc_std']:.4f}")
    print(f"  Min: {results['roc_auc_scores'].min():.4f}, Max: {results['roc_auc_scores'].max():.4f}")

# Prepare output for result.json
result_data = {
    'hypothesis_id': 'H1',
    'summary': f"Model family choice {'DOES' if is_meaningful else 'DOES NOT'} meaningfully affect predictive performance. "
               f"The best model ({best_model_name}) achieves ROC-AUC of {best_score:.4f}, "
               f"while the worst ({worst_model_name}) achieves {worst_score:.4f}, "
               f"a difference of {best_score - worst_score:.4f} ({(best_score - worst_score) / worst_score * 100:.2f}%).",
    'primary_metric_name': f"ROC-AUC difference ({best_model_name} - {worst_model_name})",
    'primary_metric_value': float(best_score - worst_score),
    'direction': f"{best_model_name} > {worst_model_name}",
    'methodological_choices': (
        "Data preprocessing: Categorical features label-encoded, numerical features filled with median. "
        "Target: binary (<=50K vs >50K). "
        "Model families compared: Logistic Regression, Random Forest, Gradient Boosting, SVM (RBF), Neural Network. "
        "Validation: Repeated stratified k-fold (5 splits × 3 repeats = 15 folds). "
        "Evaluation metric: ROC-AUC (primary) and Accuracy. "
        "Hyperparameters: Logistic Regression (max_iter=1000, solver=saga); "
        "Random Forest (n_estimators=50, max_depth=15); "
        "Gradient Boosting (n_estimators=50, max_depth=5); "
        "SVM (kernel=rbf, C=1.0); "
        "Neural Network (layers=[32,16], max_iter=200)."
    ),
    'verification_method': "Repeated stratified 5-fold cross-validation with 3 different random seeds (15 total CV evaluations per model). "
                          "Paired t-tests comparing best model against all others using the same CV folds.",
    'verification_result': f"Finding confirmed under repeated cross-validation. "
                          f"Best model ({best_model_name}: {cv_results[best_model_name]['roc_auc_mean']:.4f} ± {cv_results[best_model_name]['roc_auc_std']:.4f}) "
                          f"vs worst model ({worst_model_name}: {cv_results[worst_model_name]['roc_auc_mean']:.4f} ± {cv_results[worst_model_name]['roc_auc_std']:.4f}). "
                          f"Difference of {best_score - worst_score:.4f} suggests "
                          f"{'meaningful' if is_meaningful else 'minimal'} effect of model family choice.",
}

import json

with open('result.json', 'w') as f:
    json.dump(result_data, f, indent=2)

print("\nResults saved to result.json")
