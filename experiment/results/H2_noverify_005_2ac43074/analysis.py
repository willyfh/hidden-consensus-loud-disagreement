"""
H2: Does RandomForestClassifier() outperform LogisticRegression() on
stratified 5-fold CV ROC-AUC for the Adult Income dataset?

Both models are used with scikit-learn defaults (no hyperparameter tuning),
per the research question. Preprocessing choices (encoding, missing-value
handling, scaling) are the researcher's judgment call, documented below.
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

# Target: 1 = >50K, 0 = <=50K
y = (df["class"].str.strip() == ">50K").astype(int)
X = df.drop(columns=["class"])

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()

# Missing values appear as NaN in categorical columns (workclass, occupation,
# native-country). Impute with the most frequent category rather than
# dropping rows, to retain the full 48842-row sample. Numeric columns have
# no missing values here but are imputed with the median for robustness.
numeric_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_pipeline, numeric_cols),
    ("cat", categorical_pipeline, categorical_cols),
])

# Standard scaling is applied uniformly for both models even though
# RandomForest does not require it, so that the exact same preprocessed
# feature matrix feeds both classifiers and any performance gap reflects
# the models themselves, not differing preprocessing.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = {}
for name, clf in [
    ("logistic_regression", LogisticRegression()),
    ("random_forest", RandomForestClassifier(random_state=RANDOM_STATE)),
]:
    pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
    scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    results[name] = {
        "fold_scores": scores.tolist(),
        "mean_auc": float(scores.mean()),
        "std_auc": float(scores.std()),
    }
    print(f"{name}: mean ROC-AUC = {scores.mean():.5f} (+/- {scores.std():.5f})")
    print(f"  fold scores: {scores}")

diff = results["random_forest"]["mean_auc"] - results["logistic_regression"]["mean_auc"]
print(f"\nDifference (RF - LogReg): {diff:.5f}")

rf_wins = diff > 0
direction = "RF > LogReg" if rf_wins else ("RF < LogReg" if diff < 0 else "RF == LogReg")

summary = (
    f"Random forest achieved a mean stratified 5-fold CV ROC-AUC of "
    f"{results['random_forest']['mean_auc']:.4f} versus {results['logistic_regression']['mean_auc']:.4f} "
    f"for logistic regression, a difference of {diff:+.4f}. "
    f"{'Random forest outperforms' if rf_wins else 'Random forest does not outperform'} "
    f"logistic regression on this dataset with default hyperparameters for both models."
)

result = {
    "hypothesis_id": "H2",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(diff, 5),
    "direction": direction,
    "methodological_choices": (
        "Target encoded as binary (1 = '>50K', 0 = '<=50K'). Numeric columns "
        "(age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) "
        "median-imputed (no missing values present) and standardized with "
        "StandardScaler; categorical columns (workclass, education, "
        "marital-status, occupation, relationship, race, sex, native-country) "
        "had missing values (NaN, ~6465 cells total, concentrated in workclass, "
        "occupation, native-country) imputed with the most-frequent category "
        "rather than dropping rows, then one-hot encoded (handle_unknown='ignore'). "
        "The same preprocessing pipeline (via ColumnTransformer/Pipeline) was used "
        "for both models so the comparison isolates model choice, not preprocessing "
        "differences, even though scaling is unnecessary for RandomForest. "
        "'education' and 'education-num' were both retained as-is (redundant "
        "ordinal/nominal encodings of the same info) without deduplication. "
        "No feature selection or dimensionality reduction was applied to the "
        "one-hot-expanded feature space. Both classifiers used scikit-learn "
        "defaults (RandomForestClassifier(), LogisticRegression()) with only "
        "random_state fixed for reproducibility (RF's internal randomness); "
        "no hyperparameter tuning was performed, per the research question. "
        "Class imbalance (~24% positive class) was left unaddressed (no "
        "class_weight, no resampling), consistent with using library defaults. "
        "Validation: StratifiedKFold(n_splits=5, shuffle=True, random_state=42) "
        "via cross_val_score with scoring='roc_auc', evaluated on the full "
        "48842-row dataset (no separate held-out test set, since CV already "
        "estimates generalization for this comparison)."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
print(json.dumps(result, indent=2))
