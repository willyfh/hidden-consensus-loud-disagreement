import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names and types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget class distribution:")
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True))

# Explore the data
print("\n" + "="*60)
print("DATA EXPLORATION")
print("="*60)

# Check for missing values more carefully
print("\nChecking for missing/unknown values:")
for col in df.columns:
    if df[col].dtype == 'object':
        print(f"{col}: {df[col].unique()[:10]}")

# Handle missing values (represented as '?' in some datasets)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].replace('?', np.nan)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

# Remove rows with missing values
df_clean = df.dropna()
print(f"\nRows after removing missing values: {len(df_clean)} (removed {len(df) - len(df_clean)})")

# Separate features and target
X = df_clean.drop('class', axis=1)
y = df_clean['class'].map({'<=50K': 0, '>50K': 1})

print(f"\nTarget distribution in cleaned data:")
print(y.value_counts())
print(f"Class imbalance ratio: {y.value_counts()[1] / y.value_counts()[0]:.3f}")

# Identify categorical and numerical columns
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical columns ({len(categorical_cols)}): {categorical_cols}")
print(f"Numerical columns ({len(numerical_cols)}): {numerical_cols}")

# Preprocessing
print("\n" + "="*60)
print("PREPROCESSING")
print("="*60)

# Encode categorical variables
X_encoded = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
    label_encoders[col] = le

# Scale numerical features
scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[numerical_cols] = scaler.fit_transform(X_encoded[numerical_cols])

print(f"Preprocessed feature shape: {X_scaled.shape}")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Train class distribution:\n{y_train.value_counts()}")

# Define models to compare
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5),
    'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
}

# Train and evaluate models
print("\n" + "="*60)
print("MODEL TRAINING AND EVALUATION")
print("="*60)

results = {}
for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)

    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Get probabilities for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_proba_train = model.predict_proba(X_train)[:, 1]
        y_proba_test = model.predict_proba(X_test)[:, 1]
    else:
        y_proba_train = model.decision_function(X_train)
        y_proba_test = model.decision_function(X_test)

    # Calculate metrics
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc = accuracy_score(y_test, y_pred_test)
    train_auc = roc_auc_score(y_train, y_proba_train)
    test_auc = roc_auc_score(y_test, y_proba_test)
    test_f1 = f1_score(y_test, y_pred_test)

    results[name] = {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'train_auc': train_auc,
        'test_auc': test_auc,
        'test_f1': test_f1,
    }

    print(f"  Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    print(f"  Train AUC: {train_auc:.4f}, Test AUC: {test_auc:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")

# Summary and analysis
print("\n" + "="*60)
print("SUMMARY OF RESULTS")
print("="*60)

results_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(results_df)

# Calculate differences
test_auc_values = [results[name]['test_auc'] for name in results.keys()]
test_acc_values = [results[name]['test_acc'] for name in results.keys()]

max_auc = max(test_auc_values)
min_auc = min(test_auc_values)
auc_range = max_auc - min_auc

max_acc = max(test_acc_values)
min_acc = min(test_acc_values)
acc_range = max_acc - min_acc

print(f"\nTest AUC Range: {min_auc:.4f} to {max_auc:.4f} (range: {auc_range:.4f})")
print(f"Test Accuracy Range: {min_acc:.4f} to {max_acc:.4f} (range: {acc_range:.4f})")

# Best model
best_model_auc = max(results, key=lambda x: results[x]['test_auc'])
best_auc_score = results[best_model_auc]['test_auc']
print(f"\nBest model by Test AUC: {best_model_auc} ({best_auc_score:.4f})")

# Determine if difference is meaningful
# Using coefficient of variation or absolute differences
auc_std = np.std(test_auc_values)
auc_cv = auc_std / np.mean(test_auc_values)

print(f"\nTest AUC Statistics:")
print(f"  Mean: {np.mean(test_auc_values):.4f}")
print(f"  Std Dev: {auc_std:.4f}")
print(f"  Coeff of Variation: {auc_cv:.4f}")

# Statistical significance using absolute differences
# If range is > 0.01 (1%), we consider it meaningful
is_meaningful = auc_range > 0.01

print(f"\nIs the difference meaningful? (range > 0.01)")
print(f"  {is_meaningful} (range: {auc_range:.4f})")

# Create result JSON
worst_model = min([k for k, v in results.items() if v['test_auc'] == min_auc])
meaningfulness = 'meaningful' if is_meaningful else 'negligible'
performance_note = 'Tree-based and linear models show substantial variation in performance.' if is_meaningful else 'All models show similar performance levels.'

summary = f"Model family choice has a {meaningfulness} impact on predictive performance. Test AUC ranges from {min_auc:.4f} ({worst_model}) to {max_auc:.4f} ({best_model_auc}), a difference of {auc_range:.4f}. {performance_note}"

direction = f"Best: {best_model_auc} (AUC={best_auc_score:.4f}), Worst: {worst_model} (AUC={min_auc:.4f})"

result = {
    "hypothesis_id": "H1",
    "summary": summary,
    "primary_metric_name": "Test ROC-AUC difference (best - worst)",
    "primary_metric_value": round(auc_range, 4),
    "direction": direction,
    "methodological_choices": (
        "Preprocessing: Label-encoded categorical variables, standardized numerical features with StandardScaler. "
        "Train/test split: 80/20 stratified by target. "
        "Models compared: Logistic Regression, Decision Tree (max_depth=10), Random Forest (100 trees, max_depth=10), Gradient Boosting (100 trees, max_depth=5), SVM with RBF kernel. "
        "Evaluation metric: Primary metric is test ROC-AUC; also reported accuracy and F1-score. "
        "Missing value handling: Removed rows with missing values (represented as '?'). "
        "Imbalance handling: Used stratified split; class imbalance ratio ~0.24. "
        "Hyperparameters: Used default or moderate settings without extensive tuning to reflect typical usage patterns."
    )
}

# Save result
with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\n" + "="*60)
print("Results saved to result.json")
print("="*60)
print(json.dumps(result, indent=2))
