"""
H2: Does RandomForestClassifier (sklearn defaults) achieve higher stratified
5-fold CV ROC-AUC than LogisticRegression (sklearn defaults) on the Adult
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

# Missing values appear as NaN (already parsed by pandas from blank/NaN-like
# tokens) in workclass, occupation, native-country. Treat "missing" itself as
# an informative category rather than dropping rows, since ~7% of rows are
# affected and missingness is not random (e.g. occupation is missing exactly
# when workclass is missing).
target_col = "class"
y = (df[target_col] == ">50K").astype(int)
X = df.drop(columns=[target_col])

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
)

# Logistic regression needs scaled numeric + one-hot categorical features and
# more iterations to converge on this feature space; random forest doesn't
# need scaling but reuses the same preprocessed matrix for a fair, identical
# comparison. Both models otherwise use scikit-learn defaults as specified
# by the research question, except max_iter for LogisticRegression which is
# raised purely for convergence (default 100 fails to converge here).
logreg_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ]
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg_scores = cross_val_score(logreg_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

logreg_mean, logreg_std = logreg_scores.mean(), logreg_scores.std()
rf_mean, rf_std = rf_scores.mean(), rf_scores.std()
diff = rf_mean - logreg_mean

print("LogisticRegression per-fold ROC-AUC:", logreg_scores)
print(f"LogisticRegression mean ROC-AUC: {logreg_mean:.5f} +/- {logreg_std:.5f}")
print("RandomForest per-fold ROC-AUC:", rf_scores)
print(f"RandomForest mean ROC-AUC: {rf_mean:.5f} +/- {rf_std:.5f}")
print(f"Difference (RF - LogReg): {diff:.5f}")

# Paired t-test across the 5 folds as a lightweight significance check
from scipy import stats

t_stat, p_value = stats.ttest_rel(rf_scores, logreg_scores)
print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.5f}")

direction = "RF > LogReg" if rf_mean > logreg_mean else (
    "LogReg > RF" if logreg_mean > rf_mean else "RF == LogReg"
)

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Across stratified 5-fold CV, RandomForestClassifier achieved a mean "
        f"ROC-AUC of {rf_mean:.4f} versus {logreg_mean:.4f} for LogisticRegression "
        f"(difference of {diff:+.4f}), so the random forest {'does' if rf_mean > logreg_mean else 'does not'} "
        f"outperform logistic regression on this dataset with default hyperparameters."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": float(diff),
    "direction": direction,
    "methodological_choices": (
        "Target encoded as 1 for '>50K', 0 for '<=50K'. Missing values in workclass/"
        "occupation/native-country (NaN, ~2-6% of rows each) imputed rather than dropped: "
        "median for numeric columns (none were actually missing), constant 'Missing' category "
        "for categorical columns so missingness is treated as its own informative level. "
        "Numeric features (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) "
        "standardized with StandardScaler; categorical features one-hot encoded (handle_unknown="
        "'ignore'). Both models share the identical preprocessing pipeline via a ColumnTransformer "
        "for a fair comparison, wrapped in sklearn Pipelines so preprocessing is refit within each "
        "CV fold (no leakage). Evaluation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42), "
        "scoring='roc_auc' via cross_val_score on the full dataset (no held-out test set, per the "
        "research question's CV-only framing). Class imbalance (~24% positive) was not corrected "
        "(no class_weight, no resampling) since both models use scikit-learn defaults as specified. "
        "LogisticRegression given max_iter=1000 (default 100 does not converge on this feature space) "
        "-- this is a convergence fix, not a substantive hyperparameter change. RandomForestClassifier "
        "used entirely at default settings (100 trees, no max_depth cap). A paired t-test across the "
        "5 fold-level AUC scores was used as a lightweight significance check alongside the raw mean "
        "difference. random_state=42 fixed throughout for reproducibility."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
