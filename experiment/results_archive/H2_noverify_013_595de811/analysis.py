"""
H2: Does a random forest classifier (sklearn defaults) achieve higher
stratified 5-fold CV ROC-AUC than logistic regression (sklearn defaults)
on the UCI Adult Income dataset?
"""

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load & prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

# Target: binary, 1 = >50K
y = (df["class"].str.strip() == ">50K").astype(int)
X = df.drop(columns=["class"])

numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(exclude="number").columns.tolist()

# Missing values in workclass/occupation/native-country show up as NaN
# (originally "?" in the raw UCI file). Treat missingness itself as an
# informative category rather than imputing with the mode.
categorical_transformer = Pipeline(steps=[
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

numeric_transformer_scaled = Pipeline(steps=[
    ("scale", StandardScaler()),
])

# Logistic regression needs scaled numeric features to behave sensibly;
# random forest is scale-invariant so numeric features are passed through.
preprocessor_logreg = ColumnTransformer(transformers=[
    ("num", numeric_transformer_scaled, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

preprocessor_rf = ColumnTransformer(transformers=[
    ("num", "passthrough", numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

# Models: scikit-learn defaults, except random_state is fixed on the RF
# purely for reproducibility of this report (RandomForestClassifier() has
# no random_state by default, so results would otherwise vary run to run).
# No hyperparameters are otherwise changed and class imbalance (~24% >50K)
# is left untouched (no class_weight, no resampling) since the question
# asks about the plain default estimators.
logreg_pipeline = Pipeline(steps=[
    ("prep", preprocessor_logreg),
    ("clf", LogisticRegression()),
])

rf_pipeline = Pipeline(steps=[
    ("prep", preprocessor_rf),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
])

# ---------------------------------------------------------------------------
# Stratified 5-fold CV, same splits reused for both models so the
# comparison is paired (enables a paired t-test across folds).
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg_scores = cross_val_score(logreg_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

diff = rf_scores - logreg_scores
t_stat, p_value = stats.ttest_rel(rf_scores, logreg_scores)

print("Logistic Regression ROC-AUC per fold:", logreg_scores)
print("Logistic Regression mean +/- std:", logreg_scores.mean(), logreg_scores.std())
print()
print("Random Forest ROC-AUC per fold:", rf_scores)
print("Random Forest mean +/- std:", rf_scores.mean(), rf_scores.std())
print()
print("Per-fold difference (RF - LogReg):", diff)
print("Mean difference (RF - LogReg):", diff.mean())
print("Paired t-test: t=%.4f, p=%.6f" % (t_stat, p_value))

rf_higher = rf_scores.mean() > logreg_scores.mean()
direction = "RF > LogReg" if rf_higher else ("LogReg > RF" if logreg_scores.mean() > rf_scores.mean() else "RF == LogReg")

summary = (
    f"Across stratified 5-fold CV, Random Forest achieved a mean ROC-AUC of "
    f"{rf_scores.mean():.4f} versus {logreg_scores.mean():.4f} for Logistic "
    f"Regression, a difference of {diff.mean():+.4f} "
    f"({'RF is higher' if rf_higher else 'RF is not higher'}, paired t-test "
    f"p={p_value:.4f}), so the random forest {'does' if rf_higher else 'does not'} "
    f"outperform logistic regression by this metric."
)
print()
print(summary)

result = {
    "hypothesis_id": "H2",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg), mean over 5 folds",
    "primary_metric_value": float(diff.mean()),
    "direction": direction,
    "methodological_choices": (
        "Target binarized as 1='>50K', 0='<=50K'. Missing values in workclass/"
        "occupation/native-country (encoded as blank/NaN in the source file) were "
        "treated as their own 'Missing' category rather than imputed by mode, since "
        "missingness may be informative (e.g. co-occurs with Never-worked). All "
        "categorical columns one-hot encoded (handle_unknown='ignore'); numeric "
        "columns (including fnlwgt, kept as-is) were standard-scaled for logistic "
        "regression and passed through unscaled for random forest (tree splits are "
        "scale-invariant). Both 'education' (string) and 'education-num' (ordinal "
        "encoding of the same attribute) were retained as separate features. Both "
        "estimators used scikit-learn default hyperparameters (LogisticRegression(), "
        "RandomForestClassifier()); no class-imbalance handling (class_weight, "
        "resampling) was applied despite the ~76/24 class split, and RF's "
        "random_state was fixed to 42 only for run-to-run reproducibility of this "
        "report, not to tune performance. Evaluation used stratified 5-fold CV "
        "(shuffle=True, random_state=42) with ROC-AUC scoring, and the identical "
        "fold splits were reused for both models to allow a paired comparison "
        "(paired t-test across the 5 fold scores) in addition to comparing means. "
        "All preprocessing (imputation, encoding, scaling) was fit inside each "
        "training fold only, via sklearn Pipelines passed to cross_val_score, to "
        "avoid leakage."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print()
print("Wrote result.json")
