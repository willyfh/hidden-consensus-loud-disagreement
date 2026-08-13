"""
H2: Does RandomForestClassifier() outperform LogisticRegression() on
stratified 5-fold cross-validated ROC-AUC for the Adult Income dataset?
"""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load and prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

# Target: binary-encode class label (>50K -> 1, <=50K -> 0)
y = (df["class"].str.strip() == ">50K").astype(int)

# Drop 'education' because it is a redundant string encoding of the already
# numeric, ordinal 'education-num' column (each education level maps to one
# education-num value 1:1) -- keeping both would just duplicate a signal.
X = df.drop(columns=["class", "education"])

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
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Missing values only occur in categorical columns (workclass, occupation,
# native-country) and are coded as NaN. Treat "unknown/not reported" as its
# own category rather than imputing/dropping rows, since it may itself be
# informative (e.g. correlated with unemployment).
for col in categorical_features:
    X[col] = X[col].fillna("Missing")

# ---------------------------------------------------------------------------
# 2. Preprocessing + model pipelines
# ---------------------------------------------------------------------------
# Logistic regression needs scaled numeric features and one-hot encoded
# categoricals. Random forest is scale-invariant, but one-hot encoding with
# scaling does it no harm, so the same preprocessing pipeline is reused for
# both models to keep the comparison as apples-to-apples as possible (same
# input feature representation, only the estimator differs).
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
            categorical_features,
        ),
    ]
)

logreg_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", LogisticRegression()),  # scikit-learn defaults
    ]
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("model", RandomForestClassifier(random_state=RANDOM_STATE)),  # scikit-learn defaults (+ seed for reproducibility)
    ]
)

# ---------------------------------------------------------------------------
# 3. Stratified 5-fold cross-validated ROC-AUC
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg_scores = cross_val_score(logreg_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

logreg_mean, logreg_std = logreg_scores.mean(), logreg_scores.std()
rf_mean, rf_std = rf_scores.mean(), rf_scores.std()
diff = rf_mean - logreg_mean

print("Logistic Regression ROC-AUC per fold:", logreg_scores)
print(f"Logistic Regression ROC-AUC: {logreg_mean:.4f} +/- {logreg_std:.4f}")
print("Random Forest ROC-AUC per fold:      ", rf_scores)
print(f"Random Forest ROC-AUC:       {rf_mean:.4f} +/- {rf_std:.4f}")
print(f"Difference (RF - LogReg):    {diff:.4f}")

# ---------------------------------------------------------------------------
# 4. Write results
# ---------------------------------------------------------------------------
direction = "RF > LogReg" if diff > 0 else ("RF < LogReg" if diff < 0 else "RF == LogReg")

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Random forest achieved a mean stratified 5-fold ROC-AUC of {rf_mean:.4f} "
        f"versus {logreg_mean:.4f} for logistic regression, so random forest "
        f"{'outperforms' if diff > 0 else 'underperforms'} logistic regression by "
        f"{abs(diff):.4f} AUC points on this dataset."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": float(diff),
    "direction": direction,
    "methodological_choices": (
        "Target binarized as class=='>50K'. Dropped 'education' as a redundant "
        "string duplicate of the ordinal 'education-num' column. Missing values "
        "in workclass/occupation/native-country (categorical only; no missing "
        "numeric values) were filled with an explicit 'Missing' category rather "
        "than dropped or imputed, since missingness may itself be predictive. "
        "Numeric features standardized (StandardScaler) and categoricals "
        "one-hot encoded (OneHotEncoder, unknown categories ignored, binary "
        "columns collapsed to one dummy); the identical preprocessing pipeline "
        "was reused for both models for a fair comparison, even though RF does "
        "not require scaling. Both estimators used scikit-learn default "
        "hyperparameters (RandomForestClassifier(random_state=42) and "
        "LogisticRegression()) with no tuning; random_state fixed only for "
        "reproducibility, not as a tuned parameter. Evaluated via "
        "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) using "
        "cross_val_score with scoring='roc_auc'. No class-imbalance handling "
        "(e.g. class_weight) was applied since AUC is threshold-independent "
        "and defaults were required."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
