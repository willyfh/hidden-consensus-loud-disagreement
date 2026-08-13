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
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & inspect
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

target_col = "class"
y = (df[target_col] == ">50K").astype(int)  # 1 = minority/positive class (>50K)
X = df.drop(columns=[target_col])

categorical_cols = X.select_dtypes(include="object").columns.tolist()
# some pandas versions load these as 'str' dtype instead of 'object'
categorical_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]

# Missing values appear only in workclass, occupation, native-country (all categorical).
# Treat missing category as its own level ("Missing") rather than dropping rows,
# since dropping would discard >7% of rows and missingness may itself be informative.
for c in categorical_cols:
    X[c] = X[c].fillna("Missing")

# ---------------------------------------------------------------------------
# 2. Train/test split (stratified, held out once, used for both conditions)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. Preprocessing: one-hot encode categoricals, pass numeric columns through.
#    Fit the encoder on the training split only, then transform train/test.
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numeric_cols),
    ]
)

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

print(f"Train shape: {X_train_enc.shape}, Test shape: {X_test_enc.shape}")
print(f"Train class balance:\n{y_train.value_counts(normalize=True)}")

# ---------------------------------------------------------------------------
# 4. Condition A: No resampling — default RandomForestClassifier()
# ---------------------------------------------------------------------------
rf_plain = RandomForestClassifier(random_state=RANDOM_STATE)
rf_plain.fit(X_train_enc, y_train)
pred_plain = rf_plain.predict(X_test_enc)

f1_plain = f1_score(y_test, pred_plain, pos_label=1)
prec_plain = precision_score(y_test, pred_plain, pos_label=1)
rec_plain = recall_score(y_test, pred_plain, pos_label=1)

# ---------------------------------------------------------------------------
# 5. Condition B: SMOTE oversampling on the training data only, then the same
#    default RandomForestClassifier()
# ---------------------------------------------------------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)

print(f"Resampled train class balance:\n{y_train_res.value_counts(normalize=True)}")

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_res, y_train_res)
pred_smote = rf_smote.predict(X_test_enc)

f1_smote = f1_score(y_test, pred_smote, pos_label=1)
prec_smote = precision_score(y_test, pred_smote, pos_label=1)
rec_smote = recall_score(y_test, pred_smote, pos_label=1)

# ---------------------------------------------------------------------------
# 6. Compare
# ---------------------------------------------------------------------------
f1_diff = f1_smote - f1_plain

print("\n=== Results ===")
print(f"No resampling  -> F1(>50K): {f1_plain:.4f}  (precision={prec_plain:.4f}, recall={rec_plain:.4f})")
print(f"SMOTE          -> F1(>50K): {f1_smote:.4f}  (precision={prec_smote:.4f}, recall={rec_smote:.4f})")
print(f"F1 difference (SMOTE - no resampling): {f1_diff:.4f}")
print(f"Exceeds 0.02 threshold: {abs(f1_diff) > 0.02}")

# ---------------------------------------------------------------------------
# 7. Write result.json
# ---------------------------------------------------------------------------
exceeds = abs(f1_diff) > 0.02
if exceeds:
    direction = f"SMOTE {'increases' if f1_diff > 0 else 'decreases'} F1 by more than 0.02 ({f1_diff:+.4f})"
else:
    direction = f"No meaningful difference: SMOTE changes F1 by only {f1_diff:+.4f} (<= 0.02 threshold)"

summary = (
    f"With a default RandomForestClassifier, applying SMOTE to the training data changed the minority-class "
    f"(>50K) F1 score from {f1_plain:.4f} to {f1_smote:.4f}, a difference of {f1_diff:+.4f}, which is "
    f"{'more' if exceeds else 'not more'} than the 0.02 threshold. "
    f"{'SMOTE meaningfully affected performance' if exceeds else 'SMOTE made little practical difference'} for this classifier/dataset combination."
)

result = {
    "hypothesis_id": "H5",
    "summary": summary,
    "primary_metric_name": "F1 difference (SMOTE - no resampling), minority class >50K",
    "primary_metric_value": round(float(f1_diff), 4),
    "direction": direction,
    "methodological_choices": (
        "Target encoded as binary (1='>50K', the minority/positive class). Missing values in workclass, "
        "occupation, native-country (categorical only) imputed with an explicit 'Missing' category rather than "
        "row-dropping, to preserve ~7% of rows where missingness may be informative. Categorical features "
        "one-hot encoded (handle_unknown='ignore'); numeric features passed through unscaled (fine for "
        "tree-based RF). Single stratified 75/25 train/test split (random_state=42), evaluated once (no CV) "
        "since the question compares two fixed pipelines on held-out data. Preprocessing (encoder) fit on "
        "training split only, then applied to test split, to avoid leakage. SMOTE (imbalanced-learn default, "
        "k_neighbors=5, random_state=42) applied only to the training data after encoding/splitting (never to "
        "test data). RandomForestClassifier() used with all default hyperparameters (n_estimators=100, no "
        "class_weight adjustment) in both conditions, per the research question. F1 computed for the positive "
        "class (>50K) via sklearn f1_score(pos_label=1). No hyperparameter tuning or cross-validation was done "
        "for either condition, since the question specifies default hyperparameters and a single resampling "
        "comparison."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
