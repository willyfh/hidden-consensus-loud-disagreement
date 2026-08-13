"""
Analysis: Does model family choice meaningfully affect predictive performance?
Research Question H1
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import warnings
warnings.filterwarnings('ignore')

# Import models from different families
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

print("="*80)
print("RESEARCH QUESTION H1: Does model family choice meaningfully affect performance?")
print("="*80)

# Load data
print("\n1. LOADING AND EXPLORING DATA")
print("-" * 80)
df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"Target distribution:\n{df['class'].value_counts()}")
print(f"Missing values:\n{df.isnull().sum()}")

# Data preprocessing
print("\n2. DATA PREPROCESSING")
print("-" * 80)

# Separate features and target
X = df.drop('class', axis=1)
y = df['class'].map({'<=50K': 0, '>50K': 1})

# Handle missing values - fill with mode for categorical, mean for numeric
categorical_cols = X.select_dtypes(include=['object']).columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

for col in categorical_cols:
    X[col] = X[col].fillna(X[col].mode()[0])

for col in numeric_cols:
    X[col] = X[col].fillna(X[col].mean())

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

print(f"Features after preprocessing: {X.shape}")
print(f"Numeric features: {len(numeric_cols)}")
print(f"Categorical features (encoded): {len(categorical_cols)}")

# Train-test split for final validation
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train_full.shape[0]}, Test size: {X_test.shape[0]}")

# Define models from different families
print("\n3. BUILDING MODELS FROM DIFFERENT FAMILIES")
print("-" * 80)

models = {
    'Logistic Regression (Linear)': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree (Tree)': DecisionTreeClassifier(random_state=42, max_depth=15),
    'Random Forest (Ensemble/Tree)': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1),
    'Gradient Boosting (Ensemble/Tree)': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
    'AdaBoost (Ensemble/Tree)': AdaBoostClassifier(n_estimators=100, random_state=42),
    'K-Nearest Neighbors (Distance-based)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    'SVM (Kernel-based)': SVC(kernel='rbf', probability=True, random_state=42),
    'Naive Bayes (Probabilistic)': GaussianNB(),
    'Neural Network (Deep Learning)': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42)
}

print(f"Testing {len(models)} models from {len(set(k.split('(')[1].rstrip(')') for k in models.keys()))} different families\n")

# Cross-validation evaluation (5-fold stratified)
print("4. CROSS-VALIDATION RESULTS (5-Fold Stratified CV)")
print("-" * 80)

cv_results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"Evaluating {name}...")

    # ROC-AUC score
    auc_scores = cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='roc_auc', n_jobs=-1)

    # Accuracy score
    acc_scores = cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='accuracy', n_jobs=-1)

    cv_results[name] = {
        'auc_mean': auc_scores.mean(),
        'auc_std': auc_scores.std(),
        'auc_scores': auc_scores,
        'acc_mean': acc_scores.mean(),
        'acc_std': acc_scores.std(),
        'acc_scores': acc_scores
    }

    print(f"  ROC-AUC: {auc_scores.mean():.4f} ± {auc_scores.std():.4f}")
    print(f"  Accuracy: {acc_scores.mean():.4f} ± {acc_scores.std():.4f}\n")

# Test set evaluation
print("\n5. TEST SET EVALUATION")
print("-" * 80)

test_results = {}
for name, model in models.items():
    model.fit(X_train_full, y_train_full)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    test_results[name] = {
        'auc': auc,
        'accuracy': acc,
        'f1': f1
    }

    print(f"{name}")
    print(f"  AUC: {auc:.4f}, Accuracy: {acc:.4f}, F1: {f1:.4f}\n")

# Analysis: Is there a meaningful difference?
print("\n6. STATISTICAL ANALYSIS")
print("-" * 80)

# Calculate performance range
auc_cv_means = [cv_results[name]['auc_mean'] for name in models.keys()]
auc_cv_stds = [cv_results[name]['auc_std'] for name in models.keys()]

min_auc = min(auc_cv_means)
max_auc = max(auc_cv_means)
auc_range = max_auc - min_auc

best_model = max(models.keys(), key=lambda x: cv_results[x]['auc_mean'])
worst_model = min(models.keys(), key=lambda x: cv_results[x]['auc_mean'])

print(f"AUC Performance Range (Cross-Validation):")
print(f"  Best model:  {best_model} ({cv_results[best_model]['auc_mean']:.4f})")
print(f"  Worst model: {worst_model} ({cv_results[worst_model]['auc_mean']:.4f})")
print(f"  Difference:  {auc_range:.4f} ({auc_range*100:.2f}%)")
print(f"  Mean CV AUC: {np.mean(auc_cv_means):.4f}")
print(f"  Coefficient of Variation: {np.std(auc_cv_means)/np.mean(auc_cv_means):.4f}")

# Classification of model families
model_families = {}
for name in models.keys():
    family = name.split('(')[1].rstrip(')')
    if family not in model_families:
        model_families[family] = []
    model_families[family].append(name)

print(f"\nModel family performance (by mean CV AUC):")
family_performance = {}
for family, model_names in model_families.items():
    family_aucs = [cv_results[name]['auc_mean'] for name in model_names]
    family_performance[family] = {
        'mean': np.mean(family_aucs),
        'std': np.std(family_aucs),
        'models': model_names
    }
    print(f"  {family}: {family_performance[family]['mean']:.4f} ± {family_performance[family]['std']:.4f}")

# Calculate effect size (using coefficient of variation)
effect_size = np.std(auc_cv_means) / np.mean(auc_cv_means)
print(f"\nEffect size (CV of means): {effect_size:.4f}")

# Determine if effect is meaningful
if auc_range > 0.05:  # More than 5% difference
    meaningfulness = "YES - More than 5% difference in performance"
elif auc_range > 0.03:  # More than 3% difference
    meaningfulness = "MODERATE - 3-5% difference in performance"
else:
    meaningfulness = "NO - Less than 3% difference"

print(f"\nIs the difference meaningful? {meaningfulness}")

# Validate with repeated CV with different random seeds
print("\n7. VALIDATION: REPEATED CV WITH DIFFERENT SEEDS")
print("-" * 80)

seeds = [42, 123, 456, 789, 999]
validation_results = {name: [] for name in models.keys()}

for seed in seeds:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    for name, model in models.items():
        auc_scores = cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='roc_auc', n_jobs=-1)
        validation_results[name].append(auc_scores.mean())

# Summary of validation
print("Mean AUC across 5 repeated CV runs (different seeds):")
validation_summary = {}
for name in models.keys():
    mean = np.mean(validation_results[name])
    std = np.std(validation_results[name])
    validation_summary[name] = (mean, std)
    print(f"  {name}: {mean:.4f} ± {std:.4f}")

# Check if ranking is stable
best_model_per_seed = []
for seed in seeds:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    seed_results = {}
    for name, model in models.items():
        auc_scores = cross_val_score(model, X_train_full, y_train_full, cv=cv, scoring='roc_auc', n_jobs=-1)
        seed_results[name] = auc_scores.mean()
    best_model_per_seed.append(max(seed_results, key=seed_results.get))

print(f"\nBest model per seed: {best_model_per_seed}")
unique_best = len(set(best_model_per_seed))
print(f"Number of different best models across seeds: {unique_best}/{len(seeds)}")

# Calculate validation performance range
validation_means = [validation_summary[name][0] for name in models.keys()]
validation_range = max(validation_means) - min(validation_means)
print(f"\nValidation AUC range: {validation_range:.4f}")

print("\n" + "="*80)
print("FINAL CONCLUSION")
print("="*80)

print(f"""
Initial Finding (5-fold CV):
  - AUC range: {auc_range:.4f} ({auc_range*100:.2f}%)
  - Best vs Worst: {best_model} vs {worst_model}

Validation (Repeated CV, 5 seeds):
  - AUC range: {validation_range:.4f} ({validation_range*100:.2f}%)
  - Stability: Best model changed {unique_best} times across {len(seeds)} seeds

Effect Size: {effect_size:.4f}

ANSWER TO H1:
The choice of model family {'DOES' if auc_range > 0.03 else 'DOES NOT'} meaningfully affect
predictive performance. The performance difference across model families is
{auc_range:.4f} (or {auc_range*100:.2f}% relative difference).
""")

# Save detailed results for result.json
summary_data = {
    'initial_auc_range': float(auc_range),
    'validation_auc_range': float(validation_range),
    'best_model': best_model,
    'worst_model': worst_model,
    'best_auc': float(cv_results[best_model]['auc_mean']),
    'worst_auc': float(cv_results[worst_model]['auc_mean']),
    'effect_size': float(effect_size),
    'auc_range_percentage': float(auc_range * 100),
    'validation_results': validation_results,
    'cv_results': {name: {'auc_mean': float(results['auc_mean']),
                          'auc_std': float(results['auc_std'])}
                  for name, results in cv_results.items()}
}

import json
with open('/tmp/summary_data.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("\nAnalysis complete. Summary saved.")
