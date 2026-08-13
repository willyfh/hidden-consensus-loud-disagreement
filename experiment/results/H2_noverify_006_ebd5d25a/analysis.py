"""
H2: Does RandomForestClassifier() beat LogisticRegression() on stratified
5-fold CV ROC-AUC for the Adult Income dataset (scikit-learn defaults)?
"""

import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & inspect
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

target_col = "class"
y = (df[target_col] == ">50K").astype(int)  # positive class = >50K
X = df.drop(columns=[target_col])

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# Missing values in categorical columns (workclass, occupation, native-country
# contain NaN, originally encoded as "?") are treated as their own category
# rather than dropped or imputed with the mode, since "missingness" itself
# may carry signal (e.g. unemployed people lacking a workclass).
X[categorical_cols] = X[categorical_cols].fillna("Missing")

# ---------------------------------------------------------------------------
# 2. Preprocessing — identical for both models so the comparison isolates
#    the model class, not the feature representation. RF doesn't need scaled
#    numerics, but scaling them doesn't hurt RF and keeps the pipeline shared.
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

logreg_pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("clf", LogisticRegression()),  # scikit-learn defaults
    ]
)

rf_pipe = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),  # defaults + seed
    ]
)

# ---------------------------------------------------------------------------
# 3. Stratified 5-fold CV, ROC-AUC
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg_scores = cross_val_score(logreg_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

diff = rf_scores - logreg_scores  # per-fold paired difference, RF - LogReg
t_stat, p_value = stats.ttest_rel(rf_scores, logreg_scores)

print("LogReg fold AUCs:", logreg_scores, "mean:", logreg_scores.mean())
print("RF     fold AUCs:", rf_scores, "mean:", rf_scores.mean())
print("Per-fold diff (RF - LogReg):", diff, "mean:", diff.mean())
print(f"Paired t-test: t={t_stat:.3f}, p={p_value:.4f}")

# ---------------------------------------------------------------------------
# 4. Write results
# ---------------------------------------------------------------------------
rf_wins = bool(rf_scores.mean() > logreg_scores.mean())

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"No — random forest scored a mean 5-fold CV ROC-AUC of {rf_scores.mean():.4f}, "
        f"lower than logistic regression's {logreg_scores.mean():.4f} "
        f"(difference RF - LogReg = {diff.mean():.4f}, paired t-test p={p_value:.4f}). "
        f"Logistic regression outperforms the default random forest on this dataset."
        if not rf_wins else
        f"Yes — random forest scored a mean 5-fold CV ROC-AUC of {rf_scores.mean():.4f}, "
        f"higher than logistic regression's {logreg_scores.mean():.4f} "
        f"(difference RF - LogReg = {diff.mean():.4f}, paired t-test p={p_value:.4f})."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg), mean over 5 folds",
    "primary_metric_value": float(diff.mean()),
    "direction": "RF > LogReg" if rf_wins else "LogReg >= RF",
    "methodological_choices": (
        "Target encoded as binary (>50K=1, <=50K=0). Missing values in categorical "
        "columns (workclass, occupation, native-country, originally '?') were kept as "
        "an explicit 'Missing' category rather than imputed or dropped, since missingness "
        "may be informative. Numeric features were standardized (StandardScaler) and "
        "categorical features one-hot encoded (handle_unknown='ignore'); the identical "
        "preprocessing pipeline was used for both models so the comparison isolates model "
        "class rather than feature representation (RF doesn't require scaling, but scaling "
        "does not affect tree splits). Both models used scikit-learn default hyperparameters "
        "except random_state=42 for RandomForestClassifier (LogisticRegression is deterministic "
        "under defaults). Evaluation used StratifiedKFold(n_splits=5, shuffle=True, "
        "random_state=42) with cross_val_score(scoring='roc_auc'); preprocessing was fit "
        "within each training fold via Pipeline to avoid leakage. No class-imbalance handling "
        "(e.g. class_weight) was applied since ROC-AUC is threshold-independent and the ~24% "
        "positive rate is not extreme. Statistical significance of the paired per-fold AUC "
        "difference was assessed with a paired t-test."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
print(json.dumps(result, indent=2))
