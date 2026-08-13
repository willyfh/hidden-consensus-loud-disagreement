"""
H2: Does RandomForestClassifier() outperform LogisticRegression() on stratified
5-fold CV ROC-AUC for the Adult Income dataset?

Both models use scikit-learn defaults. Same preprocessing pipeline and same
CV folds are used for both so the comparison is paired and apples-to-apples.
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

# Target: binary-encode class label
y = (df["class"].str.strip() == ">50K").astype(int)
X = df.drop(columns=["class"])

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(exclude=["int64", "float64"]).columns.tolist()

# Missing values in categorical columns (workclass, occupation, native-country)
# are true NaNs after CSV read (originally "?" in the raw UCI data). Impute
# with an explicit "Missing" category rather than dropping ~7% of rows.
categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ]
)

# StandardScaler on numeric features is irrelevant to RandomForest (scale-
# invariant) but required for LogisticRegression to converge/behave sensibly;
# applying it to both keeps the preprocessing pipeline identical for fairness.

log_reg_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("clf", LogisticRegression()),
    ]
)

rf_pipeline = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

log_reg_scores = cross_val_score(log_reg_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipeline, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

log_reg_mean, log_reg_std = log_reg_scores.mean(), log_reg_scores.std()
rf_mean, rf_std = rf_scores.mean(), rf_scores.std()
diff = rf_mean - log_reg_mean

print("Logistic Regression ROC-AUC per fold:", log_reg_scores)
print(f"Logistic Regression ROC-AUC: {log_reg_mean:.4f} +/- {log_reg_std:.4f}")
print("Random Forest ROC-AUC per fold:      ", rf_scores)
print(f"Random Forest ROC-AUC:       {rf_mean:.4f} +/- {rf_std:.4f}")
print(f"Difference (RF - LogReg): {diff:.4f}")

direction = "RF > LogReg" if diff > 0 else ("RF < LogReg" if diff < 0 else "RF == LogReg")

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Using stratified 5-fold CV, RandomForestClassifier() achieved a mean ROC-AUC of "
        f"{rf_mean:.4f} versus {log_reg_mean:.4f} for LogisticRegression(), a difference of "
        f"{diff:+.4f}, so the random forest {'does' if diff > 0 else 'does not'} outperform "
        f"logistic regression on this dataset with default hyperparameters."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(float(diff), 4),
    "direction": direction,
    "methodological_choices": (
        "Target encoded as class=='>50K' -> 1. Missing values in categorical columns "
        "(workclass, occupation, native-country; originally '?' in raw data) imputed with an "
        "explicit 'Missing' category rather than dropping rows; numeric missing values (none "
        "present) would be median-imputed. Categorical features one-hot encoded "
        "(handle_unknown='ignore'); numeric features standard-scaled. The identical "
        "preprocessing pipeline and identical StratifiedKFold(5, shuffle=True, "
        "random_state=42) splits were used for both models to make the comparison paired and "
        "fair, even though scaling is not required for RandomForest. No class-imbalance "
        "handling applied (roughly 24%/76% class split; ROC-AUC is relatively robust to this "
        "imbalance). Both classifiers use pure scikit-learn defaults "
        "(RandomForestClassifier(), LogisticRegression()) with only random_state set for "
        "reproducibility of the RF and of the CV splitter itself, not as a hyperparameter "
        "choice. Metric: mean ROC-AUC across the 5 folds."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json:")
print(json.dumps(result, indent=2))
