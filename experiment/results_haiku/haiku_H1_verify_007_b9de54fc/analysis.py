"""
Analysis: Does model family choice meaningfully affect predictive performance?
Research Question H1
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RepeatedStratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn types and missing values:")
print(df.info())

# Data preprocessing
# Remove rows with missing target
df = df[df['class'].notna()]

# Separate features and target
X = df.drop('class', axis=1)
y = (df['class'] == '>50K').astype(int)
print(f"\nTarget distribution: {y.value_counts().to_dict()}")
print(f"Class balance: {y.mean():.2%} are >50K")

# Identify categorical and numerical columns first
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Handle missing values
for col in cat_cols:
    X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')

for col in num_cols:
    X[col] = X[col].fillna(X[col].median())

print(f"\nCategorical columns: {cat_cols}")
print(f"Numerical columns: {num_cols}")

# Encode categorical variables
le_dict = {}
X_processed = X.copy()
for col in cat_cols:
    le = LabelEncoder()
    X_processed[col] = le.fit_transform(X_processed[col])
    le_dict[col] = le

# Scale numerical features
scaler = StandardScaler()
X_processed[num_cols] = scaler.fit_transform(X_processed[num_cols])

# Train/test split for initial evaluation (70/30)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\nTrain set size: {len(X_train)}, Test set size: {len(X_test)}")

# Define models from different families
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
}

print("\n" + "="*70)
print("INITIAL PERFORMANCE ON HELD-OUT TEST SET")
print("="*70)

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    f1 = f1_score(y_test, y_pred)

    results[name] = {'accuracy': acc, 'auc': auc, 'f1': f1}
    print(f"\n{name:20s} | Accuracy: {acc:.4f} | AUC-ROC: {auc:.4f} | F1: {f1:.4f}")

# Calculate performance differences
print("\n" + "="*70)
print("PERFORMANCE DIFFERENCES (best model vs others)")
print("="*70)

# Get the best model by AUC
best_model_name = max(results, key=lambda x: results[x]['auc'])
best_auc = results[best_model_name]['auc']
print(f"\nBest performing model: {best_model_name} (AUC: {best_auc:.4f})")

differences = {}
for name in results:
    if name != best_model_name:
        auc_diff = best_auc - results[name]['auc']
        differences[name] = auc_diff
        print(f"  vs {name:25s}: {auc_diff:+.4f}")

max_auc_diff = max(abs(d) for d in differences.values())
print(f"\nMax AUC difference between models: {max_auc_diff:.4f}")

# VALIDATION: Repeated Stratified K-Fold Cross-Validation (on training set)
print("\n" + "="*70)
print("VALIDATION WITH REPEATED STRATIFIED K-FOLD CV")
print("="*70)
print("Running 3 repeats of 5-fold CV on training set...")

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
cv_results = {}

for name, model in models.items():
    print(f"  Evaluating {name}...", end=" ", flush=True)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=-1)
    cv_results[name] = cv_scores
    print(f"Done.")
    print(f"  Mean AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Range: [{cv_scores.min():.4f}, {cv_scores.max():.4f}]")

# Compare CV results
print("\n" + "="*70)
print("CV-BASED PERFORMANCE DIFFERENCES")
print("="*70)

cv_mean_aucs = {name: cv_results[name].mean() for name in models}
best_cv_model = max(cv_mean_aucs, key=cv_mean_aucs.get)
best_cv_auc = cv_mean_aucs[best_cv_model]

print(f"\nBest model by CV: {best_cv_model} (Mean AUC: {best_cv_auc:.4f})")

cv_differences = {}
for name in models:
    if name != best_cv_model:
        auc_diff = best_cv_auc - cv_mean_aucs[name]
        cv_differences[name] = auc_diff
        print(f"  vs {name:25s}: {auc_diff:+.4f}")

max_cv_auc_diff = max(abs(d) for d in cv_differences.values())
print(f"\nMax mean CV AUC difference: {max_cv_auc_diff:.4f}")

# Statistical summary
print("\n" + "="*70)
print("SUMMARY STATISTICS")
print("="*70)

print("\nTest Set AUC Scores:")
for name in sorted(results, key=lambda x: results[x]['auc'], reverse=True):
    print(f"  {name:25s}: {results[name]['auc']:.4f}")

print("\nCross-Validation Mean AUC Scores:")
for name in sorted(cv_mean_aucs, key=cv_mean_aucs.get, reverse=True):
    print(f"  {name:25s}: {cv_mean_aucs[name]:.4f}")

# Calculate coefficient of variation for each model across CV folds
print("\nModel Stability (CV std/mean):")
for name in sorted(cv_results, key=lambda x: cv_results[x].std() / cv_results[x].mean()):
    std = cv_results[name].std()
    mean = cv_results[name].mean()
    cv_coeff = std / mean
    print(f"  {name:25s}: {cv_coeff:.4f}")

# Determine if differences are meaningful
print("\n" + "="*70)
print("INTERPRETATION: Are differences meaningful?")
print("="*70)

# Threshold for "meaningful" difference (domain-dependent, but 0.01 AUC is often considered small)
meaningful_threshold = 0.01

print(f"\nUsing threshold of {meaningful_threshold:.4f} AUC for 'meaningful' difference:")
print(f"  Test set max difference: {max_auc_diff:.4f} {'(MEANINGFUL)' if max_auc_diff > meaningful_threshold else '(not meaningful)'}")
print(f"  CV max difference: {max_cv_auc_diff:.4f} {'(MEANINGFUL)' if max_cv_auc_diff > meaningful_threshold else '(not meaningful)'}")

# Compute overlap in confidence intervals
print("\nConfidence intervals (95%) from CV:")
for name in sorted(cv_mean_aucs, key=cv_mean_aucs.get, reverse=True):
    mean = cv_results[name].mean()
    std = cv_results[name].std()
    sem = std / np.sqrt(len(cv_results[name]))
    ci_lower = mean - 1.96 * sem
    ci_upper = mean + 1.96 * sem
    print(f"  {name:25s}: [{ci_lower:.4f}, {ci_upper:.4f}]")

# Find which model families show distinct performance
print("\n" + "="*70)
print("CONCLUSION")
print("="*70)

if max_cv_auc_diff > meaningful_threshold:
    print(f"\nYES: Model family DOES meaningfully affect performance.")
    print(f"Largest CV-based AUC difference: {max_cv_auc_diff:.4f}")
else:
    print(f"\nNO: Model family does NOT meaningfully affect performance.")
    print(f"Largest CV-based AUC difference: {max_cv_auc_diff:.4f}")

print(f"\nRobust metric: CV-based mean AUC range [{min(cv_mean_aucs.values()):.4f}, {max(cv_mean_aucs.values()):.4f}]")
print(f"All models evaluated on same preprocessing, train/test split methodology")
print(f"Validation: 3 repeats × 5-fold CV (15 CV runs per model)")
