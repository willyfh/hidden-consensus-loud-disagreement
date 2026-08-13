import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import json
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nData types:")
print(df.dtypes)
print("\nClass distribution:")
print(df['class'].value_counts())
print("\nClass proportions:")
print(df['class'].value_counts(normalize=True))

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum().sum())

# Separate features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode the target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)
print("\nTarget encoding:", dict(zip(le_target.classes_, le_target.transform(le_target.classes_))))

# Identify categorical and numeric columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numeric columns ({len(numeric_cols)}): {numeric_cols}")

# Encode categorical variables
X_encoded = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

# Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"\nTrain set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
print(f"Train class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

# Models to test
models = {
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
    'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

# Imbalance handling strategies
strategies = {
    'no_handling': None,
    'smote': SMOTE(random_state=42),
    'undersampling': RandomUnderSampler(random_state=42),
    'combined': ImbPipeline([
        ('undersampling', RandomUnderSampler(random_state=42)),
        ('smote', SMOTE(random_state=42))
    ])
}

# Results storage
results = {}

# Train and evaluate models
for model_name, model in models.items():
    print(f"\n{'='*60}")
    print(f"Model: {model_name}")
    print('='*60)

    results[model_name] = {}

    for strategy_name, strategy in strategies.items():
        print(f"\n  Strategy: {strategy_name}")

        # Apply imbalance handling on training data
        if strategy is None:
            X_train_processed = X_train.copy()
            y_train_processed = y_train.copy()
        elif isinstance(strategy, ImbPipeline):
            X_train_processed, y_train_processed = strategy.fit_resample(X_train, y_train)
        else:
            X_train_processed, y_train_processed = strategy.fit_resample(X_train, y_train)

        print(f"    Training set after processing: {X_train_processed.shape[0]} samples")
        print(f"    Class distribution: {np.bincount(y_train_processed)}")

        # Train model
        model_copy = type(model)(**model.get_params())
        model_copy.fit(X_train_processed, y_train_processed)

        # Predict on test set
        y_pred = model_copy.predict(X_test)
        y_pred_proba = model_copy.predict_proba(X_test)[:, 1]

        # Evaluate
        auc = roc_auc_score(y_test, y_pred_proba)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        results[model_name][strategy_name] = {
            'roc_auc': auc,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'balanced_accuracy': balanced_acc
        }

        print(f"    ROC-AUC: {auc:.4f}, F1: {f1:.4f}, Balanced Acc: {balanced_acc:.4f}")

# Analysis: Compare with and without imbalance handling
print(f"\n{'='*60}")
print("COMPARISON: Impact of Imbalance Handling")
print('='*60)

comparison_results = {}

for model_name in models.keys():
    print(f"\n{model_name}:")
    baseline = results[model_name]['no_handling']

    for strategy_name in ['smote', 'undersampling', 'combined']:
        treated = results[model_name][strategy_name]

        # Compare on multiple metrics
        auc_diff = treated['roc_auc'] - baseline['roc_auc']
        f1_diff = treated['f1'] - baseline['f1']
        balanced_acc_diff = treated['balanced_accuracy'] - baseline['balanced_accuracy']

        print(f"  {strategy_name}:")
        print(f"    ROC-AUC: {baseline['roc_auc']:.4f} → {treated['roc_auc']:.4f} (Δ {auc_diff:+.4f})")
        print(f"    F1: {baseline['f1']:.4f} → {treated['f1']:.4f} (Δ {f1_diff:+.4f})")
        print(f"    Balanced Accuracy: {baseline['balanced_accuracy']:.4f} → {treated['balanced_accuracy']:.4f} (Δ {balanced_acc_diff:+.4f})")

        comparison_results[f"{model_name}_{strategy_name}"] = {
            'roc_auc_diff': auc_diff,
            'f1_diff': f1_diff,
            'balanced_accuracy_diff': balanced_acc_diff
        }

# Summarize findings
print(f"\n{'='*60}")
print("KEY FINDINGS")
print('='*60)

all_improvements = []
for key, diffs in comparison_results.items():
    # Balanced accuracy is most appropriate for imbalanced data
    all_improvements.append((key, diffs['balanced_accuracy_diff']))

all_improvements.sort(key=lambda x: x[1], reverse=True)

print("\nImprovement in Balanced Accuracy (most relevant metric for imbalance):")
for key, imp in all_improvements[:5]:
    print(f"  {key}: {imp:+.4f}")

# Determine overall finding
avg_improvement = np.mean([imp for _, imp in all_improvements])
max_improvement = max([imp for _, imp in all_improvements])

print(f"\nAverage balanced accuracy improvement: {avg_improvement:+.4f}")
print(f"Maximum balanced accuracy improvement: {max_improvement:+.4f}")

if max_improvement > 0.01:  # >1% improvement
    finding = "Yes - addressing class imbalance improves model quality"
    direction = f"Imbalance handling improves metrics (max +{max_improvement*100:.2f}% balanced accuracy)"
else:
    finding = "Minimal/No improvement - addressing class imbalance does not substantially improve model quality"
    direction = f"Limited improvement ({max_improvement*100:.2f}% balanced accuracy)"

print(f"\nConclusion: {finding}")

# Prepare JSON result
result = {
    "hypothesis_id": "H4",
    "summary": f"Class imbalance handling strategies (SMOTE, undersampling, combined) were tested on Logistic Regression and Random Forest models. The analysis shows {'significant improvements' if max_improvement > 0.01 else 'minimal improvements'} in model quality when addressing imbalance, with the largest gain of {max_improvement*100:.2f}% in balanced accuracy.",
    "primary_metric_name": "Maximum balanced accuracy improvement (%)",
    "primary_metric_value": round(max_improvement * 100, 2),
    "direction": direction,
    "methodological_choices": "Label encoding for categorical variables; stratified 70/30 train-test split; tested Logistic Regression and Random Forest; evaluated with ROC-AUC, F1, balanced accuracy, precision, and recall; imbalance strategies: SMOTE, random undersampling, and their combination; evaluation performed on unmodified test set to avoid data leakage."
}

# Save result
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResult saved to result.json")
