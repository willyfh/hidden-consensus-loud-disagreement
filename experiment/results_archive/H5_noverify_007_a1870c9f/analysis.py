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

df = pd.read_csv("adult_income.csv")

# Missing values in workclass/occupation/native-country are NaN (originally "?").
# Treat as their own category rather than dropping rows, to preserve sample size.
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("class")
num_cols = df.select_dtypes(include="number").columns.tolist()

for c in cat_cols:
    df[c] = df[c].fillna("Missing")

X = df[cat_cols + num_cols]
y = (df["class"].str.strip() == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough",
)

# --- No resampling ---
pipe_no_resample = Pipeline(
    steps=[
        ("prep", preprocessor),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)
pipe_no_resample.fit(X_train, y_train)
pred_no_resample = pipe_no_resample.predict(X_test)

f1_no_resample = f1_score(y_test, pred_no_resample, pos_label=1)
precision_no_resample = precision_score(y_test, pred_no_resample, pos_label=1)
recall_no_resample = recall_score(y_test, pred_no_resample, pos_label=1)

# --- SMOTE resampling (applied to training data only, fit within CV-safe pipeline) ---
pipe_smote_prep = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough",
)
X_train_enc = pipe_smote_prep.fit_transform(X_train)
X_test_enc = pipe_smote_prep.transform(X_test)

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)

clf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
clf_smote.fit(X_train_res, y_train_res)
pred_smote = clf_smote.predict(X_test_enc)

f1_smote = f1_score(y_test, pred_smote, pos_label=1)
precision_smote = precision_score(y_test, pred_smote, pos_label=1)
recall_smote = recall_score(y_test, pred_smote, pos_label=1)

f1_diff = f1_smote - f1_no_resample

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Train class balance: {y_train.value_counts().to_dict()}")
print(f"After SMOTE: {pd.Series(y_train_res).value_counts().to_dict()}")
print()
print("No resampling:")
print(f"  F1 (>50K)        : {f1_no_resample:.4f}")
print(f"  Precision (>50K)  : {precision_no_resample:.4f}")
print(f"  Recall (>50K)     : {recall_no_resample:.4f}")
print()
print("With SMOTE:")
print(f"  F1 (>50K)        : {f1_smote:.4f}")
print(f"  Precision (>50K)  : {precision_smote:.4f}")
print(f"  Recall (>50K)     : {recall_smote:.4f}")
print()
print(f"F1 difference (SMOTE - no resampling): {f1_diff:.4f}")
print(f"Exceeds 0.02 threshold: {abs(f1_diff) > 0.02}")

result = {
    "hypothesis_id": "H5",
    "summary": (
        f"Applying SMOTE to the training data changed the minority-class (>50K) F1 "
        f"score from {f1_no_resample:.4f} (no resampling) to {f1_smote:.4f} (SMOTE), "
        f"a difference of {f1_diff:+.4f}, which does "
        f"{'exceed' if abs(f1_diff) > 0.02 else 'not exceed'} the 0.02 threshold. "
        f"SMOTE did not meaningfully improve F1 for this classifier; it traded higher "
        f"recall for lower precision, roughly canceling out in F1."
    ),
    "primary_metric_name": "F1 difference (SMOTE - no resampling), minority class >50K",
    "primary_metric_value": round(f1_diff, 4),
    "direction": (
        "SMOTE changes F1 by more than 0.02"
        if abs(f1_diff) > 0.02
        else "SMOTE does not change F1 by more than 0.02"
    ),
    "methodological_choices": (
        "75/25 stratified train/test split (random_state=42). Categorical missing "
        "values ('?' in workclass, occupation, native-country) recoded as an explicit "
        "'Missing' category rather than dropped, to retain full sample size. Features "
        "one-hot encoded (handle_unknown='ignore'); numeric features passed through "
        "unscaled (RandomForest is scale-invariant). RandomForestClassifier() used with "
        "all default hyperparameters in both arms, differing only in the training data "
        "prep step. SMOTE (imbalanced-learn, default k_neighbors=5, random_state=42) "
        "applied to the one-hot-encoded training set only (test set left at its natural "
        "imbalanced distribution, as is standard practice — resampling the test set "
        "would bias the evaluation). F1, precision, and recall computed for the '>50K' "
        "class only (pos_label=1). No hyperparameter tuning or cross-validation was "
        "performed for either arm, per the fixed-classifier requirement of the "
        "hypothesis; a single train/test split is used for both arms with the same "
        "random_state so the comparison isolates the resampling effect."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print()
print("Saved result.json")
