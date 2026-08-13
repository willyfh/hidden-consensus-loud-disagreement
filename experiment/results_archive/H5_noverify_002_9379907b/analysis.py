"""
H5: Does SMOTE oversampling change the minority-class (>50K) F1 score by more
than 0.02 compared to no resampling, holding the classifier fixed as a
default-hyperparameter RandomForestClassifier()?
"""

import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

# The UCI Adult dataset encodes missing values as "?" for categorical columns.
df = df.replace("?", np.nan)

target_col = "class"
y = (df[target_col].str.strip() == ">50K").astype(int)  # 1 = minority/positive class
X = df.drop(columns=[target_col])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

print("Class balance (whole dataset):")
print(y.value_counts(normalize=True))
print(f"\nCategorical columns: {cat_cols}")
print(f"Numeric columns: {num_cols}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), num_cols),
        (
            "cat",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
    ]
)

# --- Baseline: no resampling ---
pipe_baseline = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)
pipe_baseline.fit(X_train, y_train)
pred_baseline = pipe_baseline.predict(X_test)

f1_baseline = f1_score(y_test, pred_baseline, pos_label=1)
prec_baseline = precision_score(y_test, pred_baseline, pos_label=1)
rec_baseline = recall_score(y_test, pred_baseline, pos_label=1)

# --- SMOTE: oversample minority class in training data only, before fitting RF ---
# Preprocess first (SMOTE needs numeric input), then apply SMOTE to the
# transformed training matrix, then fit RandomForestClassifier() with default
# hyperparameters (matching the baseline exactly).
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_pre, y_train)

clf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
clf_smote.fit(X_train_res, y_train_res)
pred_smote = clf_smote.predict(X_test_pre)

f1_smote = f1_score(y_test, pred_smote, pos_label=1)
prec_smote = precision_score(y_test, pred_smote, pos_label=1)
rec_smote = recall_score(y_test, pred_smote, pos_label=1)

diff = f1_smote - f1_baseline

print("\n=== Results (minority class '>50K', label=1) ===")
print(f"Baseline (no resampling): F1={f1_baseline:.4f}  P={prec_baseline:.4f}  R={rec_baseline:.4f}")
print(f"SMOTE:                    F1={f1_smote:.4f}  P={prec_smote:.4f}  R={rec_smote:.4f}")
print(f"Difference (SMOTE - baseline): {diff:.4f}")
print(f"Exceeds 0.02 threshold in absolute value: {abs(diff) > 0.02}")

result = {
    "hypothesis_id": "H5",
    "summary": (
        f"SMOTE oversampling changed the minority-class (>50K) F1 score by "
        f"{diff:+.4f} (baseline F1={f1_baseline:.4f}, SMOTE F1={f1_smote:.4f}) "
        f"relative to no resampling with a default RandomForestClassifier. "
        f"{'This exceeds' if abs(diff) > 0.02 else 'This does not exceed'} the "
        f"0.02 threshold, so SMOTE {'meaningfully changed' if abs(diff) > 0.02 else 'did not meaningfully change'} "
        f"minority-class F1 in this experiment."
    ),
    "primary_metric_name": "F1 difference (SMOTE - no resampling), minority class '>50K'",
    "primary_metric_value": round(float(diff), 4),
    "direction": (
        f"SMOTE {'increases' if diff > 0 else 'decreases' if diff < 0 else 'does not change'} "
        f"minority F1 by {abs(diff):.4f}; "
        f"{'exceeds' if abs(diff) > 0.02 else 'does not exceed'} 0.02 threshold"
    ),
    "methodological_choices": (
        "Single stratified 75/25 train/test split (random_state=42), not cross-validated. "
        "Missing values ('?' tokens) imputed with median (numeric) / most-frequent (categorical) "
        "rather than dropped. Categorical features one-hot encoded (unknown categories at test "
        "time ignored). Preprocessing (impute+OHE) fit on training data only, then applied to "
        "both train and test; SMOTE applied only to the preprocessed training matrix (never to "
        "test data), with default SMOTE hyperparameters (k_neighbors=5) via imblearn. "
        "RandomForestClassifier() used with fully default hyperparameters (n_estimators=100, "
        "no class_weight, no max_depth limit) in both the no-resampling and SMOTE conditions, "
        "identical random_state=42 for the forest itself. '>50K' treated as the positive/minority "
        "class; F1 computed via sklearn f1_score with pos_label=1. No hyperparameter tuning or "
        "threshold tuning performed for either condition."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
