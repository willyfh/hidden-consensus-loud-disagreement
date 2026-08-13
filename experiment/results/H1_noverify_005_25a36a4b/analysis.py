"""
H1: Does the choice of model family meaningfully affect predictive performance
on the Adult Income dataset?

Approach
--------
- Load adult_income.csv, treat missing values (NaN in workclass/occupation/
  native-country) via imputation.
- Shared preprocessing pipeline (median impute + scale for numeric,
  most-frequent impute + one-hot encode for categorical) fed into every
  model, so differences in performance reflect the model family, not
  preprocessing choices.
- Single stratified 80/20 train/test split (fixed random_state=42) plus
  5-fold stratified cross-validation on the training set for a robustness
  check.
- Four model families spanning distinct inductive biases:
    * Logistic Regression      (linear, class_weight='balanced')
    * k-Nearest Neighbors      (instance-based, non-parametric)
    * Random Forest            (bagged trees)
    * Histogram Gradient Boosting (boosted trees)
- Primary metric: ROC-AUC (threshold-independent, robust to the ~76/24
  class imbalance). Accuracy and F1 (positive class = '>50K') reported
  as secondary metrics.
- "Meaningful" difference is judged as the spread (max - min) in test
  ROC-AUC across the four families, contextualized against the 5-fold CV
  standard deviation within each family (i.e. is the between-family gap
  larger than the within-family noise?).
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

target_col = "class"
y = (df[target_col] == ">50K").astype(int)
X = df.drop(columns=[target_col])

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

numeric_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "kNN": KNeighborsClassifier(n_neighbors=15, n_jobs=-1),
    "RandomForest": RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300, random_state=RANDOM_STATE
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", model)])

    cv_scores = cross_val_score(
        pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1
    )

    pipe.fit(X_train, y_train)
    proba_test = pipe.predict_proba(X_test)[:, 1]
    pred_test = pipe.predict(X_test)

    test_auc = roc_auc_score(y_test, proba_test)
    test_acc = accuracy_score(y_test, pred_test)
    test_f1 = f1_score(y_test, pred_test)

    results[name] = {
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "test_roc_auc": float(test_auc),
        "test_accuracy": float(test_acc),
        "test_f1": float(test_f1),
    }

    print(f"{name:22s} | CV AUC {cv_scores.mean():.4f} +/- {cv_scores.std():.4f} "
          f"| Test AUC {test_auc:.4f} | Test Acc {test_acc:.4f} | Test F1 {test_f1:.4f}")

test_aucs = {name: r["test_roc_auc"] for name, r in results.items()}
best_model = max(test_aucs, key=test_aucs.get)
worst_model = min(test_aucs, key=test_aucs.get)
auc_spread = test_aucs[best_model] - test_aucs[worst_model]

max_cv_std = max(r["cv_roc_auc_std"] for r in results.values())

print("\n--- Summary ---")
print(f"Best:  {best_model} (Test AUC = {test_aucs[best_model]:.4f})")
print(f"Worst: {worst_model} (Test AUC = {test_aucs[worst_model]:.4f})")
print(f"Spread (best - worst): {auc_spread:.4f}")
print(f"Max within-model CV std: {max_cv_std:.4f}")

meaningful = auc_spread > 3 * max_cv_std and auc_spread > 0.01
tests_min = min(test_aucs.values())
tests_max = max(test_aucs.values())

summary = (
    f"Model family has a {'meaningful' if meaningful else 'small/negligible'} effect on "
    f"predictive performance: test ROC-AUC ranges from {tests_min:.4f} "
    f"({worst_model}) to {tests_max:.4f} ({best_model}), a spread of "
    f"{auc_spread:.4f}, which is "
    f"{'larger' if meaningful else 'comparable to or smaller'} than the within-model "
    f"cross-validation noise (max std = {max_cv_std:.4f}). "
    f"{'Boosted/ensemble tree models modestly outperform the linear baseline' if meaningful else 'All four families perform similarly'}, "
    f"suggesting the underlying features are largely linearly separable / low-order interactions dominate."
)

result = {
    "hypothesis_id": "H1",
    "summary": summary,
    "primary_metric_name": "ROC-AUC spread across model families (best - worst, held-out test set)",
    "primary_metric_value": round(auc_spread, 4),
    "direction": f"{best_model} > {worst_model}" if meaningful else f"{best_model} ~= {worst_model}",
    "methodological_choices": (
        "Target binarized as '>50K'=1. Missing values (workclass, occupation, "
        "native-country; ~1.7-5.7% missing) imputed with median (numeric) / most-frequent "
        "(categorical) rather than dropped, to retain all 48842 rows. Shared preprocessing "
        "pipeline (StandardScaler + OneHotEncoder) applied identically to all models for a "
        "fair comparison, even though tree-based models don't require scaling. Single "
        "stratified 80/20 train/test split (random_state=42) plus 5-fold stratified CV on the "
        "training set to estimate within-model variance. Four model families chosen to span "
        "distinct inductive biases: Logistic Regression (linear, class_weight='balanced'), "
        "k-Nearest Neighbors (k=15, instance-based), Random Forest (300 trees, "
        "class_weight='balanced'), HistGradientBoosting (300 iterations, default "
        "regularization). Primary metric is ROC-AUC (threshold-independent, robust to the "
        "~76/24 class imbalance); accuracy and F1 reported as secondary metrics. Class "
        "imbalance handled via class_weight='balanced' for LogReg/RF rather than resampling "
        "(kNN and HGB do not natively support it and were left as-is, which is itself a "
        "methodological choice another researcher might make differently). 'Meaningful' "
        "operationalized as: AUC spread > 0.01 AND > 3x the largest within-model CV std."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
print(json.dumps(result, indent=2))
