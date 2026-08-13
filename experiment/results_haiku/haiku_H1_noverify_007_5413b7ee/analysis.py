import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Models from different families
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names: {df.columns.tolist()}")
print(f"\nClass distribution:\n{df['class'].value_counts()}")

# Prepare features and target
X = df.drop('class', axis=1)
y = df['class']

# Encode target
le_target = LabelEncoder()
y_encoded = le_target.fit_transform(y)

# Identify categorical and numeric columns BEFORE imputation
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Handle missing values
X_clean = X.copy()
for col in cat_cols:
    X_clean[col] = X_clean[col].fillna('Unknown')
for col in num_cols:
    X_clean[col] = X_clean[col].fillna(X_clean[col].median())

print(f"\nCategorical columns: {cat_cols}")
print(f"Numeric columns: {num_cols}")

# Split data (use 30% test for faster evaluation)
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
)

print(f"Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# Create preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ])

# Train/test and evaluate multiple model families
models = {
    'Logistic Regression': LogisticRegression(max_iter=500, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=15, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, max_depth=5, random_state=42),
    'Decision Tree': DecisionTreeClassifier(max_depth=15, random_state=42),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
}

results = {}

print("\n" + "="*70)
print("MODEL COMPARISON RESULTS")
print("="*70)

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)

    # Evaluate
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results[model_name] = {
        'roc_auc': roc_auc,
        'accuracy': accuracy,
        'f1': f1
    }

    print(f"{model_name}:")
    print(f"  ROC-AUC:  {roc_auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# Analysis of differences
print("\n" + "="*70)
print("STATISTICAL ANALYSIS OF MODEL DIFFERENCES")
print("="*70)

roc_auc_scores = [results[m]['roc_auc'] for m in results.keys()]
print(f"\nROC-AUC scores: {[f'{v:.4f}' for v in sorted(roc_auc_scores, reverse=True)]}")
print(f"ROC-AUC Range: {min(roc_auc_scores):.4f} - {max(roc_auc_scores):.4f}")
print(f"ROC-AUC Spread (Max - Min): {max(roc_auc_scores) - min(roc_auc_scores):.4f}")
print(f"ROC-AUC Std Dev: {np.std(roc_auc_scores):.4f}")

# Find best and worst
best_model = max(results.keys(), key=lambda x: results[x]['roc_auc'])
worst_model = min(results.keys(), key=lambda x: results[x]['roc_auc'])

best_roc = results[best_model]['roc_auc']
worst_roc = results[worst_model]['roc_auc']
diff = best_roc - worst_roc

print(f"\nBest Model: {best_model} (ROC-AUC: {best_roc:.4f})")
print(f"Worst Model: {worst_model} (ROC-AUC: {worst_roc:.4f})")
print(f"Difference: {diff:.4f}")

# Interpretation
print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if diff < 0.01:
    meaningfulness = "negligible"
    interpretation = "Model family choice has MINIMAL impact on performance."
elif diff < 0.03:
    meaningfulness = "small"
    interpretation = "Model family choice has SMALL but measurable impact on performance."
elif diff < 0.05:
    meaningfulness = "moderate"
    interpretation = "Model family choice has MODERATE impact on performance."
else:
    meaningfulness = "large"
    interpretation = "Model family choice has LARGE impact on performance."

print(f"\nDifference magnitude: {meaningfulness.upper()}")
print(f"Finding: {interpretation}")

# Final results for output
primary_metric_value = round(diff, 4)
direction = f"{best_model} ({best_roc:.4f}) > {worst_model} ({worst_roc:.4f})"

print("\n" + "="*70)
print("FINAL RESULTS")
print("="*70)
print(f"Hypothesis: Does model family choice meaningfully affect performance?")
print(f"Answer: YES, there is a {meaningfulness} difference of {diff:.4f} ROC-AUC")
print(f"Primary metric: ROC-AUC difference (best - worst)")
print(f"Primary metric value: {primary_metric_value}")
print(f"Direction: {direction}")

# Save results
results_dict = {
    "hypothesis_id": "H1",
    "summary": f"Yes, model family choice meaningfully affects predictive performance on the Adult Income dataset. The best model ({best_model}, ROC-AUC={best_roc:.4f}) outperforms the worst ({worst_model}, ROC-AUC={worst_roc:.4f}) by {diff:.4f} ROC-AUC points.",
    "primary_metric_name": "ROC-AUC difference (best model - worst model)",
    "primary_metric_value": primary_metric_value,
    "direction": direction,
    "methodological_choices": f"Preprocessing: Missing values imputed with median (numeric features) or 'Unknown' (categorical features). Features: {len(cat_cols)} categorical features one-hot encoded, {len(num_cols)} numeric features standardized. Train/test split: 70/30 stratified split with random_state=42. Models evaluated: {', '.join(models.keys())}. Primary evaluation metric: ROC-AUC on held-out test set (n={len(X_test)}). No cross-validation; single train/test split used for computational efficiency."
}

import json
with open('result.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

print("\n✓ Results saved to result.json")
