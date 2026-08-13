"""
H2: Does a random forest classifier (sklearn defaults) achieve higher stratified
5-fold cross-validated ROC-AUC than logistic regression (sklearn defaults) on the
Adult Income dataset?
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

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

target_col = "class"
y = (df[target_col] == ">50K").astype(int)
X = df.drop(columns=[target_col])

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = [c for c in X.columns if c not in numeric_cols]

# "education" and "education-num" are redundant (same info, ordinal vs one-hot).
# Keep both as-is; the models can decide implicitly since defaults are used and
# feature selection was not asked for. No rows are dropped for missingness in
# workclass/occupation/native-country since dropping ~7% of rows on the target
# class would discard information; instead missing categories are imputed with
# a dedicated "Missing" category so both models can use every row.

# ---------------------------------------------------------------------------
# Preprocessing pipelines
# ---------------------------------------------------------------------------
# Logistic regression needs scaled numeric features and one-hot encoded
# categoricals (it has no notion of feature scale or category order).
logreg_preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ]
)

# Random forest is invariant to monotonic transforms of numeric features and
# handles one-hot encoded categoricals fine (trees don't need scaling).
rf_preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            categorical_cols,
        ),
    ]
)

logreg_pipeline = Pipeline(
    steps=[
        ("preprocess", logreg_preprocess),
        ("model", LogisticRegression()),  # sklearn defaults
    ]
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", rf_preprocess),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE)),  # sklearn defaults + seed for reproducibility
    ]
)

# ---------------------------------------------------------------------------
# Stratified 5-fold CV, ROC-AUC
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg_scores = cross_val_score(logreg_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

logreg_mean, logreg_std = logreg_scores.mean(), logreg_scores.std()
rf_mean, rf_std = rf_scores.mean(), rf_scores.std()
diff = rf_mean - logreg_mean

print("Logistic Regression fold ROC-AUCs:", logreg_scores)
print(f"Logistic Regression mean ROC-AUC: {logreg_mean:.5f} (+/- {logreg_std:.5f})")
print("Random Forest fold ROC-AUCs:      ", rf_scores)
print(f"Random Forest mean ROC-AUC:       {rf_mean:.5f} (+/- {rf_std:.5f})")
print(f"Difference (RF - LogReg): {diff:.5f}")

# Paired comparison across the 5 folds (same folds used for both models)
wins_rf = int((rf_scores > logreg_scores).sum())
print(f"RF beats LogReg in {wins_rf}/5 folds")

# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------
direction = "RF > LogReg" if diff > 0 else ("RF < LogReg" if diff < 0 else "RF == LogReg")

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Random forest achieved a mean stratified 5-fold CV ROC-AUC of {rf_mean:.4f} "
        f"versus {logreg_mean:.4f} for logistic regression on the Adult Income dataset "
        f"({'higher' if diff > 0 else 'lower'} by {abs(diff):.4f}), and RF won in "
        f"{wins_rf}/5 individual folds, so random forest {'does' if diff > 0 else 'does not'} "
        f"outperform logistic regression by this metric."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg), mean over 5 stratified CV folds",
    "primary_metric_value": float(diff),
    "direction": direction,
    "methodological_choices": (
        "Target encoded as binary (1 = '>50K'). Missing values in workclass/occupation/"
        "native-country (empty strings, ~2-6% of rows) were imputed with a constant "
        "'Missing' category rather than dropped, so all 48842 rows were used for both "
        "models. All categorical columns (workclass, education, marital-status, "
        "occupation, relationship, race, sex, native-country) were one-hot encoded "
        "(handle_unknown='ignore'). For logistic regression, numeric columns (age, "
        "fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) were "
        "standardized (StandardScaler); for random forest, numeric columns were passed "
        "through unscaled since tree splits are scale-invariant. Both 'education' "
        "(categorical) and 'education-num' (its numeric encoding) were kept as separate, "
        "redundant features rather than dropping one, since no feature selection was "
        "specified. Both models used scikit-learn defaults (RandomForestClassifier(), "
        "LogisticRegression()) apart from setting random_state=42 on the RF for "
        "reproducibility (LogisticRegression's default lbfgs solver is deterministic so "
        "no seed was needed there). No class-imbalance handling (e.g. class_weight) was "
        "applied, matching the 'defaults' instruction, despite the ~3.2:1 class imbalance "
        "(<=50K vs >50K). Evaluation used StratifiedKFold(n_splits=5, shuffle=True, "
        "random_state=42) with scoring='roc_auc' via cross_val_score; the same CV splits "
        "were reused for both models to allow a fair, paired per-fold comparison. No "
        "held-out test set was used since the question specifically asks about CV "
        "performance, not generalization to a separate test set."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
