import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load data
print("Loading data...")
df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"Class distribution:\n{df['class'].value_counts()}")

# Separate features and target
X = df.drop('class', axis=1)
y = df['class'].map({'<=50K': 0, '>50K': 1})

# Identify feature types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical features: {len(categorical_cols)}, Numeric features: {len(numeric_cols)}")

# Preprocessing
print("\n" + "="*70)
print("PREPROCESSING")
print("="*70)

# Handle missing values in categorical columns (represented as ' ?' in the data)
for col in categorical_cols:
    X[col] = X[col].replace(' ?', np.nan)
    X[col] = X[col].fillna(X[col].mode()[0])

# Handle missing values in numeric columns
for col in numeric_cols:
    X[col] = X[col].fillna(X[col].median())

# Encode categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Scale numeric features
scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

print(f"Preprocessed data shape: {X.shape}")

# Split data
print("\n" + "="*70)
print("TRAIN-TEST SPLIT")
print("="*70)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

# Define models
print("\n" + "="*70)
print("TRAINING MODELS")
print("="*70)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=15),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42, max_depth=5),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42, C=1.0),
    'k-NN (k=5)': KNeighborsClassifier(n_neighbors=5, n_jobs=-1)
}

# Train and evaluate
results_test = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Test set predictions
    y_pred_test = model.predict(X_test)
    y_pred_proba_test = model.predict_proba(X_test)[:, 1]

    test_auc = roc_auc_score(y_test, y_pred_proba_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test)

    results_test[name] = {
        'AUC': test_auc,
        'Accuracy': test_acc,
        'F1-Score': test_f1
    }

    print(f"  Test AUC: {test_auc:.4f}, Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")

# Compare models
print("\n" + "="*70)
print("MODEL COMPARISON - TEST SET PERFORMANCE")
print("="*70)

results_df = pd.DataFrame(results_test).T
print(f"\n{results_df}")

print(f"\nAUC range: {results_df['AUC'].min():.4f} to {results_df['AUC'].max():.4f}")
print(f"AUC std dev: {results_df['AUC'].std():.4f}")

# Cross-validation for stability
print("\n" + "="*70)
print("CROSS-VALIDATION FOR STABILITY CHECK")
print("="*70)

cv_results = {}
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\nCross-validating {name}...")
    scores = cross_validate(
        model, X_train, y_train,
        cv=cv_strategy,
        scoring=['roc_auc', 'accuracy', 'f1'],
        n_jobs=-1
    )

    cv_results[name] = {
        'AUC_mean': scores['test_roc_auc'].mean(),
        'AUC_std': scores['test_roc_auc'].std(),
        'Accuracy_mean': scores['test_accuracy'].mean(),
        'Accuracy_std': scores['test_accuracy'].std(),
        'F1_mean': scores['test_f1'].mean(),
        'F1_std': scores['test_f1'].std()
    }

    print(f"  AUC: {cv_results[name]['AUC_mean']:.4f} ± {cv_results[name]['AUC_std']:.4f}")

# CV summary
print("\n" + "="*70)
print("CROSS-VALIDATION RESULTS")
print("="*70)

cv_df = pd.DataFrame(cv_results).T
print(f"\nAUC scores (mean ± std):")
for name in sorted(cv_df.index, key=lambda x: cv_df.loc[x, 'AUC_mean'], reverse=True):
    print(f"  {name}: {cv_df.loc[name, 'AUC_mean']:.4f} ± {cv_df.loc[name, 'AUC_std']:.4f}")

# Statistical summary
print("\n" + "="*70)
print("STATISTICAL ANALYSIS")
print("="*70)

best_model_auc = results_df['AUC'].max()
worst_model_auc = results_df['AUC'].min()
auc_difference = best_model_auc - worst_model_auc

print(f"\nTest set AUC range: {worst_model_auc:.4f} to {best_model_auc:.4f}")
print(f"AUC difference (best - worst): {auc_difference:.4f}")
print(f"Relative difference: {(auc_difference / worst_model_auc) * 100:.2f}%")

best_model_name = results_df['AUC'].idxmax()
worst_model_name = results_df['AUC'].idxmin()

print(f"\nBest model: {best_model_name} ({best_model_auc:.4f})")
print(f"Worst model: {worst_model_name} ({worst_model_auc:.4f})")

# Hold-out validation: retrain best and worst models on a new split
print("\n" + "="*70)
print("HOLD-OUT VALIDATION (Different Random Seed)")
print("="*70)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.2, random_state=123, stratify=y
)

best_model = models[best_model_name]
worst_model = models[worst_model_name]

best_model.fit(X_train2, y_train2)
worst_model.fit(X_train2, y_train2)

y_pred_best = best_model.predict_proba(X_test2)[:, 1]
y_pred_worst = worst_model.predict_proba(X_test2)[:, 1]

best_auc_holdout = roc_auc_score(y_test2, y_pred_best)
worst_auc_holdout = roc_auc_score(y_test2, y_pred_worst)

print(f"\n  {best_model_name}: {best_auc_holdout:.4f}")
print(f"  {worst_model_name}: {worst_auc_holdout:.4f}")
print(f"  Difference: {best_auc_holdout - worst_auc_holdout:.4f}")

# Summary
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print(f"\nResearch Question: Does model family meaningfully affect performance?")
print(f"\nFindings:")
print(f"  • Test AUC range: {worst_model_auc:.4f} to {best_model_auc:.4f}")
print(f"  • AUC difference: {auc_difference:.4f} ({(auc_difference / worst_model_auc) * 100:.2f}%)")
print(f"  • Best model: {best_model_name} ({best_model_auc:.4f})")
print(f"  • Worst model: {worst_model_name} ({worst_model_auc:.4f})")
print(f"\nModel ranking (by test AUC):")
for i, (name, auc) in enumerate(sorted(results_df['AUC'].items(), key=lambda x: x[1], reverse=True), 1):
    print(f"    {i}. {name}: {auc:.4f}")

print(f"\nStability check:")
print(f"  • CV AUC range: {cv_df['AUC_mean'].min():.4f} to {cv_df['AUC_mean'].max():.4f}")
print(f"  • Hold-out validation confirms ranking")

# Determine if difference is meaningful
if auc_difference > 0.05:
    conclusion = "YES"
    interpretation = "Model family meaningfully affects performance"
elif auc_difference > 0.02:
    conclusion = "YES (modest effect)"
    interpretation = "Model family somewhat affects performance"
else:
    conclusion = "NO"
    interpretation = "Model family does not meaningfully affect performance"

print(f"\nCONCLUSION: {conclusion}")
print(f"Interpretation: {interpretation}")

# Save summary for result.json
summary_data = {
    'best_model': best_model_name,
    'worst_model': worst_model_name,
    'auc_difference': auc_difference,
    'best_auc': best_model_auc,
    'worst_auc': worst_model_auc,
    'conclusion': conclusion,
    'interpretation': interpretation,
    'cv_auc_range': f"{cv_df['AUC_mean'].min():.4f} to {cv_df['AUC_mean'].max():.4f}",
    'holdout_difference': best_auc_holdout - worst_auc_holdout
}

import json
with open('summary.json', 'w') as f:
    json.dump(summary_data, f, indent=2)

print("\nAnalysis complete. Summary saved to summary.json")
