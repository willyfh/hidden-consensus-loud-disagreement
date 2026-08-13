"""
H2: Does RandomForestClassifier() outperform LogisticRegression() (both sklearn
defaults) on stratified 5-fold CV ROC-AUC for predicting `class` in the Adult
Income dataset?
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

y = (df["class"] == ">50K").astype(int)
X = df.drop(columns=["class"])

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

# Missing values appear only in categorical columns (workclass, occupation,
# native-country), encoded as NaN (originally "?"). Impute with a distinct
# "Missing" category rather than dropping rows, to keep all 48842 records.
categorical_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

numeric_pipeline = Pipeline([
    ("scale", StandardScaler()),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = {}
for name, clf in [
    ("logreg", LogisticRegression(random_state=RANDOM_STATE, max_iter=1000)),
    ("rf", RandomForestClassifier(random_state=RANDOM_STATE)),
]:
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    results[name] = scores
    print(f"{name}: fold AUCs = {np.round(scores, 4)}, mean = {scores.mean():.4f}, std = {scores.std():.4f}")

diff = results["rf"].mean() - results["logreg"].mean()
print(f"\nRF - LogReg mean ROC-AUC diff = {diff:.4f}")

# Paired t-test across the 5 folds to gauge whether the difference is
# consistent/meaningful relative to fold-to-fold variability.
from scipy import stats
t_stat, p_value = stats.ttest_rel(results["rf"], results["logreg"])
print(f"Paired t-test: t = {t_stat:.4f}, p = {p_value:.4f}")

direction = "RF > LogReg" if diff > 0 else ("RF < LogReg" if diff < 0 else "RF == LogReg")

summary = (
    f"Random forest achieved a mean stratified 5-fold ROC-AUC of {results['rf'].mean():.4f} "
    f"versus {results['logreg'].mean():.4f} for logistic regression (both with default "
    f"hyperparameters), a difference of {diff:+.4f} in favor of "
    f"{'random forest' if diff > 0 else 'logistic regression'} "
    f"(paired t-test p={p_value:.4f})."
)
print("\n" + summary)

result = {
    "hypothesis_id": "H2",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(float(diff), 4),
    "direction": direction,
    "methodological_choices": (
        "Encoded target as class=='>50K'. Numeric features (age, fnlwgt, education-num, "
        "capital-gain, capital-loss, hours-per-week) standardized (StandardScaler); this "
        "matters for LogisticRegression convergence/regularization but is a no-op for "
        "RandomForest. Categorical features (workclass, education, marital-status, "
        "occupation, relationship, race, sex, native-country) one-hot encoded "
        "(OneHotEncoder, handle_unknown='ignore'). Missing values (present only in "
        "workclass, occupation, native-country, originally '?') imputed as an explicit "
        "'Missing' category rather than row-deletion, to retain the full 48842 rows and "
        "let both models use missingness as a signal. Both classifiers used pure sklearn "
        "defaults (LogisticRegression(max_iter=1000) — max_iter raised only to avoid a "
        "convergence warning, not to change the estimator's decision boundary; "
        "RandomForestClassifier() untouched). Evaluation: StratifiedKFold(n_splits=5, "
        "shuffle=True, random_state=42) with cross_val_score(scoring='roc_auc'), same "
        "folds for both models to allow a paired comparison. Class imbalance (~24% "
        "positive) was not corrected via resampling/class_weight since ROC-AUC is "
        "relatively robust to it and the question specifies scikit-learn defaults. "
        "Significance assessed with a paired t-test across the 5 fold scores."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
