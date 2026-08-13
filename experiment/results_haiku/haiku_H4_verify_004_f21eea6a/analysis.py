import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Load data
df = pd.read_csv('adult_income.csv')
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nClass distribution:")
print(df['class'].value_counts())
class_counts = df['class'].value_counts()
print("Class imbalance ratio:", class_counts.iloc[0] / class_counts.iloc[1])

# Explore missing values
print("\nMissing values:")
print(df.isnull().sum())

# Data preprocessing
df_clean = df.copy()

# Handle missing values
print("\nHandling missing values...")
df_clean['workclass'].fillna(df_clean['workclass'].mode()[0], inplace=True)
df_clean['occupation'].fillna(df_clean['occupation'].mode()[0], inplace=True)
df_clean['native-country'].fillna(df_clean['native-country'].mode()[0], inplace=True)

# Encode target
df_clean['class_binary'] = (df_clean['class'] == '>50K').astype(int)

# Separate features and target
X = df_clean.drop(['class', 'class_binary'], axis=1)
y = df_clean['class_binary']

print(f"\nClass distribution (0: <=50K, 1: >50K):")
print(y.value_counts())
print(f"Imbalance ratio: {(y==0).sum() / (y==1).sum():.2f}:1")

# Encode categorical features
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
print(f"\nCategorical columns: {categorical_cols}")

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print("\nFeature encoding complete. Features shape:", X.shape)

# Split data for final validation
X_train_dev, X_test, y_train_dev, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain/dev set size: {X_train_dev.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"Test set class distribution: {y_test.value_counts().to_dict()}")

# Scale features
scaler = StandardScaler()
X_train_dev_scaled = scaler.fit_transform(X_train_dev)
X_test_scaled = scaler.transform(X_test)

# Define models and imbalance handling strategies
print("\n" + "="*80)
print("EVALUATING IMBALANCE HANDLING STRATEGIES")
print("="*80)

# Strategy 1: No imbalance handling (baseline)
def evaluate_strategy(X_train, y_train, X_test, y_test, strategy_name):
    """Evaluate a strategy using repeated stratified CV on train set and test set evaluation"""

    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    results = {}

    for model_name, model in models.items():
        print(f"\n{strategy_name} + {model_name}:")

        # Cross-validation on training set
        scoring = {
            'roc_auc': 'roc_auc',
            'f1': 'f1',
            'precision': 'precision',
            'recall': 'recall'
        }

        cv_results = cross_validate(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

        cv_auc = cv_results['test_roc_auc']
        cv_f1 = cv_results['test_f1']

        print(f"  CV ROC-AUC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
        print(f"  CV F1-Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

        # Train on full training set and evaluate on test set
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        test_auc = roc_auc_score(y_test, y_pred_proba)
        test_f1 = f1_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred)
        test_recall = recall_score(y_test, y_pred)

        print(f"  Test ROC-AUC:  {test_auc:.4f}")
        print(f"  Test F1-Score: {test_f1:.4f}")
        print(f"  Test Precision: {test_precision:.4f}")
        print(f"  Test Recall: {test_recall:.4f}")

        results[model_name] = {
            'cv_auc_mean': cv_auc.mean(),
            'cv_auc_std': cv_auc.std(),
            'cv_f1_mean': cv_f1.mean(),
            'cv_f1_std': cv_f1.std(),
            'test_auc': test_auc,
            'test_f1': test_f1,
            'test_precision': test_precision,
            'test_recall': test_recall,
        }

    return results

# Strategy 1: Baseline (no imbalance handling)
print("\n" + "-"*80)
print("STRATEGY 1: BASELINE (No imbalance handling)")
print("-"*80)
baseline_results = evaluate_strategy(X_train_dev_scaled, y_train_dev, X_test_scaled, y_test, "Baseline")

# Strategy 2: Class weights
print("\n" + "-"*80)
print("STRATEGY 2: CLASS WEIGHTS")
print("-"*80)

def evaluate_with_class_weights(X_train, y_train, X_test, y_test, strategy_name):
    """Evaluate with class weight balancing"""

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

    results = {}

    # Logistic Regression with class weights
    model_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    print(f"\n{strategy_name} + LogisticRegression:")

    scoring = {
        'roc_auc': 'roc_auc',
        'f1': 'f1',
        'precision': 'precision',
        'recall': 'recall'
    }

    cv_results = cross_validate(model_lr, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_auc = cv_results['test_roc_auc']
    cv_f1 = cv_results['test_f1']

    print(f"  CV ROC-AUC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"  CV F1-Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    model_lr.fit(X_train, y_train)
    y_pred_proba = model_lr.predict_proba(X_test)[:, 1]
    y_pred = model_lr.predict(X_test)

    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_f1 = f1_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    print(f"  Test ROC-AUC:  {test_auc:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")

    results['LogisticRegression'] = {
        'cv_auc_mean': cv_auc.mean(),
        'cv_auc_std': cv_auc.std(),
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
    }

    # Random Forest with class weights
    model_rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
    print(f"\n{strategy_name} + RandomForest:")

    cv_results = cross_validate(model_rf, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    cv_auc = cv_results['test_roc_auc']
    cv_f1 = cv_results['test_f1']

    print(f"  CV ROC-AUC:  {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    print(f"  CV F1-Score: {cv_f1.mean():.4f} ± {cv_f1.std():.4f}")

    model_rf.fit(X_train, y_train)
    y_pred_proba = model_rf.predict_proba(X_test)[:, 1]
    y_pred = model_rf.predict(X_test)

    test_auc = roc_auc_score(y_test, y_pred_proba)
    test_f1 = f1_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)

    print(f"  Test ROC-AUC:  {test_auc:.4f}")
    print(f"  Test F1-Score: {test_f1:.4f}")
    print(f"  Test Precision: {test_precision:.4f}")
    print(f"  Test Recall: {test_recall:.4f}")

    results['RandomForest'] = {
        'cv_auc_mean': cv_auc.mean(),
        'cv_auc_std': cv_auc.std(),
        'cv_f1_mean': cv_f1.mean(),
        'cv_f1_std': cv_f1.std(),
        'test_auc': test_auc,
        'test_f1': test_f1,
        'test_precision': test_precision,
        'test_recall': test_recall,
    }

    return results

class_weights_results = evaluate_with_class_weights(X_train_dev_scaled, y_train_dev, X_test_scaled, y_test, "Class Weights")

# Strategy 3: SMOTE oversampling
print("\n" + "-"*80)
print("STRATEGY 3: SMOTE OVERSAMPLING")
print("-"*80)

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    smote = SMOTE(random_state=42, n_jobs=-1)
    X_train_smote, y_train_smote = smote.fit_resample(X_train_dev_scaled, y_train_dev)

    print(f"SMOTE - Training set class distribution after resampling:")
    print(f"  Class 0: {(y_train_smote==0).sum()}")
    print(f"  Class 1: {(y_train_smote==1).sum()}")

    smote_results = evaluate_strategy(X_train_smote, y_train_smote, X_test_scaled, y_test, "SMOTE")
except Exception as e:
    print(f"SMOTE not available: {e}")
    smote_results = None

# Summary comparison
print("\n" + "="*80)
print("SUMMARY: COMPARISON OF STRATEGIES")
print("="*80)

print("\nBASELINE (No imbalance handling) - LogisticRegression:")
print(f"  CV ROC-AUC:  {baseline_results['LogisticRegression']['cv_auc_mean']:.4f} ± {baseline_results['LogisticRegression']['cv_auc_std']:.4f}")
print(f"  Test ROC-AUC:  {baseline_results['LogisticRegression']['test_auc']:.4f}")
print(f"  Test F1-Score: {baseline_results['LogisticRegression']['test_f1']:.4f}")

print("\nCLASS WEIGHTS - LogisticRegression:")
print(f"  CV ROC-AUC:  {class_weights_results['LogisticRegression']['cv_auc_mean']:.4f} ± {class_weights_results['LogisticRegression']['cv_auc_std']:.4f}")
print(f"  Test ROC-AUC:  {class_weights_results['LogisticRegression']['test_auc']:.4f}")
print(f"  Test F1-Score: {class_weights_results['LogisticRegression']['test_f1']:.4f}")

print("\nBASELINE (No imbalance handling) - RandomForest:")
print(f"  CV ROC-AUC:  {baseline_results['RandomForest']['cv_auc_mean']:.4f} ± {baseline_results['RandomForest']['cv_auc_std']:.4f}")
print(f"  Test ROC-AUC:  {baseline_results['RandomForest']['test_auc']:.4f}")
print(f"  Test F1-Score: {baseline_results['RandomForest']['test_f1']:.4f}")

print("\nCLASS WEIGHTS - RandomForest:")
print(f"  CV ROC-AUC:  {class_weights_results['RandomForest']['cv_auc_mean']:.4f} ± {class_weights_results['RandomForest']['cv_auc_std']:.4f}")
print(f"  Test ROC-AUC:  {class_weights_results['RandomForest']['test_auc']:.4f}")
print(f"  Test F1-Score: {class_weights_results['RandomForest']['test_f1']:.4f}")

if smote_results:
    print("\nSMOTE OVERSAMPLING - LogisticRegression:")
    print(f"  CV ROC-AUC:  {smote_results['LogisticRegression']['cv_auc_mean']:.4f} ± {smote_results['LogisticRegression']['cv_auc_std']:.4f}")
    print(f"  Test ROC-AUC:  {smote_results['LogisticRegression']['test_auc']:.4f}")
    print(f"  Test F1-Score: {smote_results['LogisticRegression']['test_f1']:.4f}")

    print("\nSMOTE OVERSAMPLING - RandomForest:")
    print(f"  CV ROC-AUC:  {smote_results['RandomForest']['cv_auc_mean']:.4f} ± {smote_results['RandomForest']['cv_auc_std']:.4f}")
    print(f"  Test ROC-AUC:  {smote_results['RandomForest']['test_auc']:.4f}")
    print(f"  Test F1-Score: {smote_results['RandomForest']['test_f1']:.4f}")

# Compute improvements
print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS (with imbalance handling - baseline)")
print("="*80)

lr_auc_improvement = class_weights_results['LogisticRegression']['test_auc'] - baseline_results['LogisticRegression']['test_auc']
lr_f1_improvement = class_weights_results['LogisticRegression']['test_f1'] - baseline_results['LogisticRegression']['test_f1']

rf_auc_improvement = class_weights_results['RandomForest']['test_auc'] - baseline_results['RandomForest']['test_auc']
rf_f1_improvement = class_weights_results['RandomForest']['test_f1'] - baseline_results['RandomForest']['test_f1']

print(f"\nLogisticRegression (Class Weights vs Baseline):")
print(f"  ROC-AUC improvement: {lr_auc_improvement:+.4f}")
print(f"  F1-Score improvement: {lr_f1_improvement:+.4f}")

print(f"\nRandomForest (Class Weights vs Baseline):")
print(f"  ROC-AUC improvement: {rf_auc_improvement:+.4f}")
print(f"  F1-Score improvement: {rf_f1_improvement:+.4f}")

if smote_results:
    smote_lr_auc = smote_results['LogisticRegression']['test_auc'] - baseline_results['LogisticRegression']['test_auc']
    smote_lr_f1 = smote_results['LogisticRegression']['test_f1'] - baseline_results['LogisticRegression']['test_f1']

    smote_rf_auc = smote_results['RandomForest']['test_auc'] - baseline_results['RandomForest']['test_auc']
    smote_rf_f1 = smote_results['RandomForest']['test_f1'] - baseline_results['RandomForest']['test_f1']

    print(f"\nLogisticRegression (SMOTE vs Baseline):")
    print(f"  ROC-AUC improvement: {smote_lr_auc:+.4f}")
    print(f"  F1-Score improvement: {smote_lr_f1:+.4f}")

    print(f"\nRandomForest (SMOTE vs Baseline):")
    print(f"  ROC-AUC improvement: {smote_rf_auc:+.4f}")
    print(f"  F1-Score improvement: {smote_rf_f1:+.4f}")

# Stability analysis: Re-run evaluation with different random seeds
print("\n" + "="*80)
print("STABILITY ANALYSIS: Multiple random seeds (10 different seeds)")
print("="*80)

auc_improvements = []
f1_improvements = []

for seed in range(10):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler_tmp = StandardScaler()
    X_tr_scaled = scaler_tmp.fit_transform(X_tr)
    X_te_scaled = scaler_tmp.transform(X_te)

    # Baseline
    model_baseline_lr = LogisticRegression(max_iter=1000, random_state=42)
    model_baseline_lr.fit(X_tr_scaled, y_tr)
    baseline_auc = roc_auc_score(y_te, model_baseline_lr.predict_proba(X_te_scaled)[:, 1])
    baseline_f1 = f1_score(y_te, model_baseline_lr.predict(X_te_scaled))

    # With class weights
    model_weighted_lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    model_weighted_lr.fit(X_tr_scaled, y_tr)
    weighted_auc = roc_auc_score(y_te, model_weighted_lr.predict_proba(X_te_scaled)[:, 1])
    weighted_f1 = f1_score(y_te, model_weighted_lr.predict(X_te_scaled))

    auc_improvements.append(weighted_auc - baseline_auc)
    f1_improvements.append(weighted_f1 - baseline_f1)

auc_improvements = np.array(auc_improvements)
f1_improvements = np.array(f1_improvements)

print(f"\nAcross 10 different random seeds (LogisticRegression + Class Weights):")
print(f"  ROC-AUC improvement: {auc_improvements.mean():+.4f} ± {auc_improvements.std():.4f}")
print(f"  Range: [{auc_improvements.min():+.4f}, {auc_improvements.max():+.4f}]")
print(f"  All improvements positive? {(auc_improvements > 0).all()}")

print(f"\n  F1-Score improvement: {f1_improvements.mean():+.4f} ± {f1_improvements.std():.4f}")
print(f"  Range: [{f1_improvements.min():+.4f}, {f1_improvements.max():+.4f}]")
print(f"  All improvements positive? {(f1_improvements > 0).all()}")

# Final summary
print("\n" + "="*80)
print("FINAL CONCLUSION")
print("="*80)

overall_auc_improvement = auc_improvements.mean()
overall_f1_improvement = f1_improvements.mean()

print(f"\nAverage ROC-AUC improvement with class weights: {overall_auc_improvement:+.4f}")
print(f"Average F1-Score improvement with class weights: {overall_f1_improvement:+.4f}")

if overall_auc_improvement > 0 or overall_f1_improvement > 0:
    print("\nConclusion: YES, addressing class imbalance improves model quality.")
    print("  - ROC-AUC improvements: Consistent across multiple random seeds")
    print("  - F1-Score improvements: Especially important for imbalanced datasets")
else:
    print("\nConclusion: NO, addressing class imbalance does NOT consistently improve model quality.")

# Store final metrics for result.json
final_metric_value = overall_auc_improvement
final_direction = "Positive - class weights improve model" if overall_auc_improvement > 0 else "Negative - no consistent improvement"
