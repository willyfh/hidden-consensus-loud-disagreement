"""
H2: Does RandomForestClassifier() outperform LogisticRegression() on stratified
5-fold CV ROC-AUC for predicting `class` (<=50K vs >50K) in the Adult dataset?

Both models are used with scikit-learn defaults (per the research question),
except for random_state (for reproducibility) and n_jobs (for speed) on RF,
and a raised max_iter for LogisticRegression so it actually converges on the
one-hot-encoded feature space rather than silently returning an unconverged fit.
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

target = "class"
y = (df[target] == ">50K").astype(int)
X = df.drop(columns=[target])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# Missing values in workclass/occupation/native-country are left as NaN by
# pandas; treat "unknown" as its own category rather than dropping rows,
# since ~7% of rows have at least one missing categorical.
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])

preprocess = ColumnTransformer([
    ("cat", cat_pipe, cat_cols),
    ("num", num_pipe, num_cols),
])

# RF doesn't need scaling but a shared preprocessed matrix keeps the
# comparison simple and doesn't hurt RF performance.
rf_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
])

logreg_pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
logreg_scores = cross_val_score(logreg_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

rf_mean, rf_std = rf_scores.mean(), rf_scores.std()
logreg_mean, logreg_std = logreg_scores.mean(), logreg_scores.std()
diff = rf_mean - logreg_mean

print("RF per-fold ROC-AUC:     ", np.round(rf_scores, 5))
print("LogReg per-fold ROC-AUC: ", np.round(logreg_scores, 5))
print(f"RF mean ROC-AUC:      {rf_mean:.5f} +/- {rf_std:.5f}")
print(f"LogReg mean ROC-AUC:  {logreg_mean:.5f} +/- {logreg_std:.5f}")
print(f"Difference (RF - LogReg): {diff:.5f}")

direction = "RF > LogReg" if diff > 0 else ("RF < LogReg" if diff < 0 else "RF == LogReg")

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Random forest achieved a mean stratified 5-fold CV ROC-AUC of {rf_mean:.4f} "
        f"versus {logreg_mean:.4f} for logistic regression "
        f"({'higher' if diff > 0 else 'lower'} by {abs(diff):.4f}), so with default "
        f"hyperparameters {'random forest does' if diff > 0 else 'random forest does not'} "
        f"outperform logistic regression on this dataset."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": float(diff),
    "direction": direction,
    "methodological_choices": (
        "Target encoded as binary (1 = '>50K'). Missing values in workclass/occupation/"
        "native-country (~2-6% each) imputed as a distinct 'Missing' category rather than "
        "dropped, to retain all 48842 rows. Categorical features one-hot encoded "
        "(handle_unknown='ignore'); numeric features median-imputed (no missing values "
        "present) and standardized. The same preprocessed feature matrix (one-hot + scaled) "
        "was used for both models for a like-for-like comparison, even though RF does not "
        "require scaling. Both classifiers used scikit-learn default hyperparameters as "
        "specified, except LogisticRegression max_iter was raised to 1000 (default 100) "
        "since it failed to converge on the expanded one-hot feature space otherwise; "
        "random_state=42 fixed for reproducibility. Evaluation: stratified 5-fold CV "
        "(shuffle=True, random_state=42), scoring='roc_auc' via cross_val_score, no held-out "
        "test set since the question asks specifically about CV performance. Class imbalance "
        "(~24% positive) was not explicitly corrected (no class_weight, no resampling) since "
        "the question specifies scikit-learn defaults for both models."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
