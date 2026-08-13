"""
H2: Does RandomForestClassifier() outperform LogisticRegression() on stratified
5-fold CV ROC-AUC for the Adult Income dataset (scikit-learn defaults)?
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

# Target: 1 = >50K (the minority, "positive" class of interest)
y = (df["class"] == ">50K").astype(int)
X = df.drop(columns=["class"])

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

# Missing values in workclass/occupation/native-country are NaN (not the literal
# "?" string). Impute categoricals with a constant "Missing" category rather than
# dropping rows, since missingness itself may be informative (e.g. unemployed).
numeric_pipe = Pipeline([
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, numeric_cols),
    ("cat", categorical_pipe, categorical_cols),
])

# StandardScaler is a no-op for tree splits in RandomForest, so using the same
# preprocessing pipeline for both models keeps the comparison apples-to-apples
# without disadvantaging LogisticRegression (which does need scaling).
logreg_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression()),
])

rf_pipe = Pipeline([
    ("prep", preprocessor),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg_scores = cross_val_score(logreg_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

logreg_mean, logreg_std = logreg_scores.mean(), logreg_scores.std()
rf_mean, rf_std = rf_scores.mean(), rf_scores.std()
diff = rf_mean - logreg_mean

print("LogisticRegression per-fold ROC-AUC:", logreg_scores)
print(f"LogisticRegression mean ROC-AUC: {logreg_mean:.5f} (+/- {logreg_std:.5f})")
print("RandomForest per-fold ROC-AUC:", rf_scores)
print(f"RandomForest mean ROC-AUC: {rf_mean:.5f} (+/- {rf_std:.5f})")
print(f"Difference (RF - LogReg): {diff:.5f}")

# Paired t-test across the 5 fold scores (same folds for both models) as a quick
# check that the difference isn't just fold-to-fold noise.
from scipy import stats
t_stat, p_value = stats.ttest_rel(rf_scores, logreg_scores)
print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.5f}")

direction = "RF > LogReg" if diff > 0 else ("RF < LogReg" if diff < 0 else "RF == LogReg")

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Random forest achieved a mean 5-fold stratified CV ROC-AUC of {rf_mean:.4f} "
        f"versus {logreg_mean:.4f} for logistic regression on the Adult Income dataset, "
        f"so {'random forest outperforms' if diff > 0 else 'logistic regression outperforms'} "
        f"logistic regression with default hyperparameters (paired t-test p={p_value:.4f})."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": float(diff),
    "direction": direction,
    "methodological_choices": (
        "Target encoded as 1 for '>50K' (minority class, 24% of rows). Missing values "
        "in workclass/occupation/native-country (stored as NaN, not literal '?') were "
        "imputed as a constant 'Missing' category rather than dropped, preserving all "
        "48842 rows. Categorical features one-hot encoded (handle_unknown='ignore'); "
        "numeric features standardized with StandardScaler. The identical preprocessing "
        "pipeline (including scaling) was used for both models via a shared "
        "ColumnTransformer to keep the comparison fair, even though scaling has no effect "
        "on RandomForest splits. Both models used scikit-learn defaults "
        "(RandomForestClassifier(), LogisticRegression()) with only random_state fixed "
        "for reproducibility of the RF and of fold assignment. Evaluation: "
        "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='roc_auc' "
        "via cross_val_score, mean AUC across folds compared. A paired t-test across the "
        "5 matched fold scores was used to gauge whether the difference exceeds fold-to-fold "
        "noise. No hyperparameter tuning was performed for either model, per the research "
        "question's specification of 'scikit-learn defaults'."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
