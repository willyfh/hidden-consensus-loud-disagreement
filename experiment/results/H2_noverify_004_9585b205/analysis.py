"""
H2: Does RandomForestClassifier() outperform LogisticRegression() on stratified
5-fold CV ROC-AUC for the Adult Income dataset (predicting class <=50K vs >50K)?
"""

import json

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

df = pd.read_csv("adult_income.csv")

# Missing values in this dataset are encoded as literal "?" strings in the
# original UCI source, but here (OpenML variant) they already come through
# as NaN for workclass, occupation, native-country. Treat them as a
# "Missing" category rather than dropping rows, so no information is lost.
for col in ["workclass", "occupation", "native-country"]:
    df[col] = df[col].fillna("Missing")

y = (df["class"] == ">50K").astype(int)
X = df.drop(columns=["class"])

# fnlwgt is a census sampling weight, not a real demographic feature of the
# individual; it is kept in as a numeric predictor since no instruction says
# to exclude it, but noted here as a judgment call.
numeric_features = [
    "age",
    "fnlwgt",
    "education-num",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
]
categorical_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# education-num is a numeric encoding of the ordinal `education` column, so
# both are included but this is a known redundancy left in for both models
# equally (a fair-comparison choice, not a leakage issue).

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# Same preprocessing pipeline (one-hot + scaling) used for both models so
# the comparison isolates the classifier, not the encoding scheme. Scaling
# is a no-op for RandomForest's split logic but harmless.

logreg_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", LogisticRegression()),
])

rf_pipeline = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg_scores = cross_val_score(logreg_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

logreg_mean, logreg_std = logreg_scores.mean(), logreg_scores.std()
rf_mean, rf_std = rf_scores.mean(), rf_scores.std()
diff = rf_mean - logreg_mean

print("LogisticRegression fold AUCs:", logreg_scores)
print(f"LogisticRegression mean ROC-AUC: {logreg_mean:.5f} (+/- {logreg_std:.5f})")
print("RandomForest fold AUCs:", rf_scores)
print(f"RandomForest mean ROC-AUC:       {rf_mean:.5f} (+/- {rf_std:.5f})")
print(f"Difference (RF - LogReg): {diff:.5f}")

# Paired t-test across the 5 fold scores as a lightweight significance check
from scipy import stats
t_stat, p_value = stats.ttest_rel(rf_scores, logreg_scores)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.5f}")

direction = "RF > LogReg" if diff > 0 else ("RF < LogReg" if diff < 0 else "RF == LogReg")

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Random forest achieved a higher mean 5-fold stratified CV ROC-AUC "
        f"({rf_mean:.4f}) than logistic regression ({logreg_mean:.4f}) on the "
        f"Adult Income dataset, a difference of {diff:.4f} "
        f"(paired t-test p={p_value:.4f})."
        if diff > 0 else
        f"Random forest did not achieve a higher mean 5-fold stratified CV "
        f"ROC-AUC ({rf_mean:.4f}) than logistic regression ({logreg_mean:.4f}) "
        f"on the Adult Income dataset, a difference of {diff:.4f} "
        f"(paired t-test p={p_value:.4f})."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(float(diff), 5),
    "direction": direction,
    "methodological_choices": (
        "Target encoded as class==\">50K\"=1. Missing values in workclass/occupation/"
        "native-country (blank in source) imputed as a 'Missing' category rather than "
        "row-dropping. Numeric features (age, fnlwgt, education-num, capital-gain, "
        "capital-loss, hours-per-week) median-imputed and standard-scaled; categorical "
        "features one-hot encoded (handle_unknown='ignore'). Identical preprocessing "
        "pipeline used for both models so the comparison isolates classifier choice, "
        "not encoding scheme (scaling is a no-op for RF but included for parity). "
        "fnlwgt (census sampling weight) and education-num (redundant with education) "
        "kept in as features for both models. Both models used with scikit-learn "
        "default hyperparameters as specified: RandomForestClassifier() (100 trees) "
        "and LogisticRegression() (L2 penalty, C=1.0, lbfgs solver, may not fully "
        "converge on one-hot data within default max_iter but left as default since "
        "not otherwise specified). Evaluation: StratifiedKFold(n_splits=5, shuffle=True, "
        "random_state=42), scoring='roc_auc' via cross_val_score. Statistical check: "
        "paired t-test across the 5 fold AUC scores. No class-imbalance handling "
        "applied (~24% positive class) beyond stratified folds, per default settings "
        "requested in the research question."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
