"""
H2: Does RandomForestClassifier (sklearn defaults) achieve higher stratified
5-fold CV ROC-AUC than LogisticRegression (sklearn defaults) on the Adult
Income dataset?

Methodology:
- Load adult_income.csv, treat '?' as missing.
- Target: class (<=50K / >50K) -> binary 0/1, positive class = >50K.
- Features: all other columns. Numeric columns used as-is (imputed with
  median if missing, though none expected). Categorical columns
  one-hot encoded (this benefits LogisticRegression; RF could also use
  ordinal/native categorical handling but we keep preprocessing identical
  for both models for a fair, simple comparison).
- Missing categorical values ('?') imputed with the most frequent category.
- LogisticRegression needs feature scaling for numeric columns to converge
  reasonably; StandardScaler applied to numeric features. RandomForest is
  scale-invariant so scaling doesn't hurt it - using the same pipeline
  structure (with scaling included) keeps preprocessing identical/fair
  across both models, isolating the model choice as the only difference.
- Both models otherwise use scikit-learn defaults (RandomForestClassifier(),
  LogisticRegression()), except random_state is set on RF for reproducibility
  (does not affect defaults' behavior, only reproducibility) and
  LogisticRegression's default max_iter=100 is left as-is (default), which
  may raise a ConvergenceWarning but is intentionally left as the "default"
  configuration per the research question.
- Evaluation: Stratified 5-fold CV, scoring = roc_auc, using the full
  dataset (no separate holdout) since the question specifically asks about
  cross-validated ROC-AUC.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv", na_values="?", skipinitialspace=True)

# Basic cleanup: strip whitespace from string columns (OpenML/UCI adult data
# often has leading spaces), normalize target labels (some versions have
# trailing '.' e.g. '>50K.')
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()

df["class"] = df["class"].str.rstrip(".")

y = (df["class"] == ">50K").astype(int)
X = df.drop(columns=["class"])

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include="object").columns.tolist()

numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

rf_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
])

logreg_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", LogisticRegression()),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    rf_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    logreg_scores = cross_val_score(logreg_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

rf_mean, rf_std = rf_scores.mean(), rf_scores.std()
logreg_mean, logreg_std = logreg_scores.mean(), logreg_scores.std()
diff = rf_mean - logreg_mean

print("RandomForest CV ROC-AUC scores:", rf_scores)
print(f"RandomForest mean ROC-AUC: {rf_mean:.5f} (+/- {rf_std:.5f})")
print("LogisticRegression CV ROC-AUC scores:", logreg_scores)
print(f"LogisticRegression mean ROC-AUC: {logreg_mean:.5f} (+/- {logreg_std:.5f})")
print(f"Difference (RF - LogReg): {diff:.5f}")

direction = "RF > LogReg" if diff > 0 else ("RF < LogReg" if diff < 0 else "RF == LogReg")

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Random forest achieved a mean stratified 5-fold CV ROC-AUC of {rf_mean:.4f} "
        f"versus {logreg_mean:.4f} for logistic regression on the Adult Income dataset, "
        f"so random forest {'outperforms' if diff > 0 else 'does not outperform'} logistic "
        f"regression by {abs(diff):.4f} AUC points with default hyperparameters."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(float(diff), 5),
    "direction": direction,
    "methodological_choices": (
        "Target binarized as class=='>50K' (positive). Numeric features median-imputed "
        "and standardized; categorical features ('?' treated as missing) most-frequent-"
        "imputed and one-hot encoded (handle_unknown='ignore'); identical preprocessing "
        "pipeline used for both models to isolate model choice as the only difference "
        "(scaling doesn't affect RF but ensures fairness/consistency). Both models used "
        "scikit-learn default hyperparameters (RandomForestClassifier(), "
        "LogisticRegression()), aside from random_state=42 for reproducibility of RF and "
        "the CV splitter. Evaluated via StratifiedKFold(n_splits=5, shuffle=True, "
        "random_state=42) cross_val_score with scoring='roc_auc' on the full dataset "
        "(no held-out test set), since the research question is specifically about "
        "cross-validated performance. No class-imbalance correction applied (class_weight "
        "left at sklearn default of None for LogisticRegression), consistent with using "
        "'scikit-learn defaults' as specified in the question."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json:")
print(json.dumps(result, indent=2))
