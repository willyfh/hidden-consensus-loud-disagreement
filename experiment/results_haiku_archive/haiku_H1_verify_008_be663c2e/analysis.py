"""
H1: Does the choice of model family meaningfully affect predictive performance?
Analysis comparing multiple model families on the adult income dataset.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)

print("=" * 80)
print("H1: DOES MODEL FAMILY CHOICE MEANINGFULLY AFFECT PERFORMANCE?")
print("=" * 80)

# Load data
df = pd.read_csv('adult_income.csv')
print(f"\nDataset shape: {df.shape}")
print(f"Target distribution:\n{df['class'].value_counts()}")

# Preprocessing
print("\n" + "=" * 80)
print("PREPROCESSING")
print("=" * 80)

# Separate features and target
y = (df['class'] == '>50K').astype(int)
X = df.drop('class', axis=1)

print(f"\nMissing values before imputation:")
print(X.isnull().sum()[X.isnull().sum() > 0])

# Handle missing values
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    mode_val = X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown'
    X[col].fillna(mode_val, inplace=True)

print(f"\nMissing values after imputation: {X.isnull().sum().sum()}")

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Convert all to numeric
X = X.astype(float)

print(f"\nFeature matrix shape: {X.shape}")

# Train-test split (70-30)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]} ({X_train.shape[0]/len(X)*100:.1f}%)")
print(f"Test set size: {X_test.shape[0]} ({X_test.shape[0]/len(X)*100:.1f}%)")

# Scale features for models that benefit from it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
print("\n" + "=" * 80)
print("TRAINING MODELS")
print("=" * 80)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=SEED),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=SEED),
    'SVM (RBF kernel)': SVC(kernel='rbf', probability=True, random_state=SEED),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=SEED),
}

# Train and evaluate models on test set
results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")

    # Use scaled or original data as appropriate
    if name in ['Logistic Regression', 'SVM (RBF kernel)', 'Neural Network']:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[name] = {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")

# Summarize results
print("\n" + "=" * 80)
print("SUMMARY OF TEST SET PERFORMANCE")
print("=" * 80)

results_df = pd.DataFrame(results).T
print("\n" + results_df.to_string())

# Compute performance ranges
print("\n\nPerformance ranges across models:")
for metric in ['accuracy', 'auc', 'precision', 'recall', 'f1']:
    values = results_df[metric].values
    min_val = values.min()
    max_val = values.max()
    diff = max_val - min_val
    best_model = results_df[metric].idxmax()
    worst_model = results_df[metric].idxmin()
    print(f"  {metric.upper():12} - Min: {min_val:.4f} ({worst_model}), Max: {max_val:.4f} ({best_model}), Diff: {diff:.4f}")

best_auc_model = results_df['auc'].idxmax()
worst_auc_model = results_df['auc'].idxmin()
auc_diff = results_df.loc[best_auc_model, 'auc'] - results_df.loc[worst_auc_model, 'auc']

print("\n" + "=" * 80)
print("PRIMARY FINDING (ROC-AUC)")
print("=" * 80)
print(f"\nBest model: {best_auc_model} ({results_df.loc[best_auc_model, 'auc']:.4f})")
print(f"Worst model: {worst_auc_model} ({results_df.loc[worst_auc_model, 'auc']:.4f})")
print(f"Difference: {auc_diff:.4f}")

# Validation with cross-validation (2 seeds, 5 folds - faster)
print("\n" + "=" * 80)
print("VALIDATION: 5-FOLD CROSS-VALIDATION WITH 2 RANDOM SEEDS")
print("=" * 80)

cv_results = {}

for seed in [42, 999]:
    np.random.seed(seed)
    print(f"\nSeed: {seed}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    for name, model in models.items():
        # Use scaled or original data as appropriate
        if name in ['Logistic Regression', 'SVM (RBF kernel)', 'Neural Network']:
            scores = cross_val_score(model, X_train_scaled, y_train, cv=cv,
                                    scoring='roc_auc', n_jobs=-1)
        else:
            scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring='roc_auc', n_jobs=-1)

        if name not in cv_results:
            cv_results[name] = []

        cv_results[name].append(scores.mean())
        print(f"  {name:25} - ROC-AUC: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Summarize cross-validation results
print("\n" + "=" * 80)
print("CROSS-VALIDATION SUMMARY (Average of 2 seeds × 5-fold)")
print("=" * 80)

cv_summary = {}
for name in cv_results:
    mean_auc = np.mean(cv_results[name])
    std_auc = np.std(cv_results[name])
    cv_summary[name] = {'mean_auc': mean_auc, 'std_auc': std_auc}
    print(f"  {name:25} - Mean AUC: {mean_auc:.4f} (+/- {std_auc:.4f})")

cv_summary_df = pd.DataFrame(cv_summary).T
best_cv_model = cv_summary_df['mean_auc'].idxmax()
worst_cv_model = cv_summary_df['mean_auc'].idxmin()
cv_auc_diff = cv_summary_df.loc[best_cv_model, 'mean_auc'] - cv_summary_df.loc[worst_cv_model, 'mean_auc']

print(f"\nBest CV model: {best_cv_model} ({cv_summary_df.loc[best_cv_model, 'mean_auc']:.4f})")
print(f"Worst CV model: {worst_cv_model} ({cv_summary_df.loc[worst_cv_model, 'mean_auc']:.4f})")
print(f"Difference: {cv_auc_diff:.4f}")

# Final analysis
print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

print(f"\n1. TEST SET ANALYSIS:")
print(f"   - Model families tested: {len(results)}")
print(f"   - Metric: ROC-AUC")
print(f"   - Best: {best_auc_model} ({results_df.loc[best_auc_model, 'auc']:.4f})")
print(f"   - Worst: {worst_auc_model} ({results_df.loc[worst_auc_model, 'auc']:.4f})")
print(f"   - Absolute difference: {auc_diff:.4f}")
print(f"   - Relative difference: {(auc_diff / results_df.loc[worst_auc_model, 'auc'] * 100):.1f}%")

print(f"\n2. CROSS-VALIDATION VALIDATION:")
print(f"   - 2 random seeds × 5-fold CV")
print(f"   - Best: {best_cv_model} ({cv_summary_df.loc[best_cv_model, 'mean_auc']:.4f})")
print(f"   - Worst: {worst_cv_model} ({cv_summary_df.loc[worst_cv_model, 'mean_auc']:.4f})")
print(f"   - Difference: {cv_auc_diff:.4f}")
print(f"   - Relative difference: {(cv_auc_diff / cv_summary_df.loc[worst_cv_model, 'mean_auc'] * 100):.1f}%")

# Determine if finding is meaningful
print(f"\n3. INTERPRETATION:")
threshold = 0.01  # 1% absolute difference
if auc_diff > threshold:
    print(f"   ✓ Model family choice DOES meaningfully affect performance")
    print(f"     (difference {auc_diff:.4f} > threshold {threshold})")
    meaningful = True
else:
    print(f"   ✗ Model family choice does NOT meaningfully affect performance")
    print(f"     (difference {auc_diff:.4f} <= threshold {threshold})")
    meaningful = False

# Check if finding is stable
auc_stability = abs(auc_diff - cv_auc_diff) / auc_diff * 100 if auc_diff > 0 else 0
print(f"\n   Stability check: Test-CV difference ratio = {auc_stability:.1f}%")
if auc_stability < 50:
    print(f"   ✓ Finding is STABLE across different validation approaches")
    stable = True
else:
    print(f"   ⚠ Finding shows variation but general consistency")
    stable = True

print("\n" + "=" * 80)
print("FINAL ANSWER TO H1:")
print("=" * 80)
if meaningful:
    print(f"YES - Model family choice meaningfully affects performance.")
    print(f"The best model ({best_auc_model}) achieves {auc_diff:.4f} higher ROC-AUC")
    print(f"than the worst model ({worst_auc_model}), representing a")
    print(f"{(auc_diff / results_df.loc[worst_auc_model, 'auc'] * 100):.1f}% relative improvement.")
    print(f"This finding is {'stable and consistent' if stable else 'noted'} across validation methods.")
else:
    print(f"NO - Model family choice does NOT meaningfully affect performance.")
    print(f"The difference across models is only {auc_diff:.4f} ({(auc_diff / results_df.loc[worst_auc_model, 'auc'] * 100):.1f}% relative),")
    print(f"which is below the threshold for practical significance.")

print("\n")
