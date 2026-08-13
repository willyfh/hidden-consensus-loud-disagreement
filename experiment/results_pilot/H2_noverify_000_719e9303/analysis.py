"""
H2: Does RandomForestClassifier() beat LogisticRegression() on stratified
5-fold CV ROC-AUC for the Adult Income dataset?
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

# Target: 1 = >50K, 0 = <=50K
y = (df["class"].str.strip() == ">50K").astype(int)

# 'education' is a redundant string version of the already-numeric
# 'education-num' -> drop to avoid duplicating the same signal twice.
X = df.drop(columns=["class", "education"])

numeric_features = [
    "age", "fnlwgt", "education-num",
    "capital-gain", "capital-loss", "hours-per-week",
]
categorical_features = [
    "workclass", "marital-status", "occupation",
    "relationship", "race", "sex", "native-country",
]

# Missing values in workclass/occupation/native-country are left as NaN by
# pandas (source used '?'); impute as an explicit "Missing" category rather
# than dropping rows, so no data is discarded.
for c in categorical_features:
    X[c] = X[c].astype("object").where(X[c].notna(), "Missing")

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)

logreg_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])

rf_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    logreg_scores = cross_val_score(logreg_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

logreg_mean, logreg_std = logreg_scores.mean(), logreg_scores.std()
rf_mean, rf_std = rf_scores.mean(), rf_scores.std()
diff = rf_mean - logreg_mean

print("LogisticRegression per-fold ROC-AUC:", logreg_scores)
print(f"LogisticRegression mean +/- std: {logreg_mean:.5f} +/- {logreg_std:.5f}")
print("RandomForest per-fold ROC-AUC:", rf_scores)
print(f"RandomForest mean +/- std: {rf_mean:.5f} +/- {rf_std:.5f}")
print(f"Difference (RF - LogReg): {diff:.5f}")

# Paired comparison across the 5 folds (same splits used for both models)
from scipy import stats
t_stat, p_value = stats.ttest_rel(rf_scores, logreg_scores)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.5f}")

comparison_verb = "outperforms" if diff > 0 else ("underperforms" if diff < 0 else "ties")
significance = "statistically significant" if p_value < 0.05 else "not statistically significant"
result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Random forest (mean ROC-AUC={rf_mean:.4f}) {comparison_verb} logistic "
        f"regression (mean ROC-AUC={logreg_mean:.4f}) under stratified 5-fold "
        f"CV, a difference of {diff:+.4f} AUC points that is {significance} "
        f"(paired t-test p={p_value:.3f})."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": float(diff),
    "direction": "RF > LogReg" if diff > 0 else ("RF < LogReg" if diff < 0 else "RF == LogReg"),
    "methodological_choices": (
        "Target encoded as 1 for '>50K', 0 for '<=50K'. Dropped 'education' "
        "column as redundant with numeric 'education-num'. Missing values in "
        "workclass/occupation/native-country (originally '?') imputed as an "
        "explicit 'Missing' category rather than dropping rows. Numeric "
        "features standardized (StandardScaler) and categorical features "
        "one-hot encoded (handle_unknown='ignore'); identical preprocessing "
        "pipeline used for both models for a fair comparison, even though "
        "scaling is not required for RandomForest. Both classifiers used "
        "scikit-learn defaults except LogisticRegression max_iter raised from "
        "100 to 1000 (only to ensure solver convergence on the one-hot-"
        "encoded feature space; default C=1.0, solver='lbfgs' unchanged). "
        "Evaluation via StratifiedKFold(n_splits=5, shuffle=True, "
        "random_state=42) with scoring='roc_auc'. No class-imbalance "
        "correction applied (kept both models at sklearn defaults, i.e. no "
        "class_weight='balanced'). fnlwgt retained as a numeric feature "
        "though it is a census sampling weight rather than a demographic "
        "attribute. A paired t-test across the 5 folds was used to check "
        "whether RF's advantage is consistent rather than noise."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
