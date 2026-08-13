"""
H5: Does applying SMOTE oversampling to the training data change the
minority-class (>50K) F1 score by more than 0.02 compared to no resampling,
holding the classifier fixed as a default-hyperparameter RandomForestClassifier()?
"""

import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

# Treat literal "?" as missing, then drop rows with missing values (small
# fraction of the data, concentrated in workclass/occupation/native-country).
df = df.replace("?", np.nan)
df = df.dropna()

target = "class"
y = (df[target] == ">50K").astype(int)
X = df.drop(columns=[target])

# fnlwgt is a census sampling weight, not a real demographic feature -- drop it.
X = X.drop(columns=["fnlwgt"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough",
)

# --- Baseline: no resampling ---
baseline_pipe = Pipeline(
    steps=[
        ("prep", preprocess),
        ("clf", RandomForestClassifier()),
    ]
)
baseline_pipe.fit(X_train, y_train)
y_pred_baseline = baseline_pipe.predict(X_test)
f1_baseline = f1_score(y_test, y_pred_baseline, pos_label=1)

# --- SMOTE on training data, same classifier ---
X_train_enc = preprocess.fit_transform(X_train, y_train)
# OneHotEncoder + passthrough may yield a sparse matrix; SMOTE needs dense array.
if hasattr(X_train_enc, "toarray"):
    X_train_enc = X_train_enc.toarray()

smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)

X_test_enc = preprocess.transform(X_test)
if hasattr(X_test_enc, "toarray"):
    X_test_enc = X_test_enc.toarray()

rf_smote = RandomForestClassifier()
rf_smote.fit(X_train_res, y_train_res)
y_pred_smote = rf_smote.predict(X_test_enc)
f1_smote = f1_score(y_test, y_pred_smote, pos_label=1)

diff = f1_smote - f1_baseline

print("Class balance (train):", y_train.value_counts().to_dict())
print("Class balance (train, after SMOTE):", pd.Series(y_train_res).value_counts().to_dict())
print()
print("=== Baseline (no resampling) ===")
print(classification_report(y_test, y_pred_baseline, target_names=["<=50K", ">50K"]))
print("F1 (>50K):", f1_baseline)
print()
print("=== SMOTE ===")
print(classification_report(y_test, y_pred_smote, target_names=["<=50K", ">50K"]))
print("F1 (>50K):", f1_smote)
print()
print("Difference (SMOTE - baseline):", diff)
print("Exceeds 0.02 threshold:", abs(diff) > 0.02)

result = {
    "hypothesis_id": "H5",
    "summary": (
        f"Applying SMOTE to the training data changed the >50K F1 score from "
        f"{f1_baseline:.4f} (no resampling) to {f1_smote:.4f} (SMOTE), a difference of "
        f"{diff:+.4f}, which does {'exceed' if abs(diff) > 0.02 else 'NOT exceed'} the "
        f"0.02 threshold. SMOTE did not meaningfully improve, and slightly "
        f"{'hurt' if diff < 0 else 'helped'}, minority-class F1 for the default random forest."
    ),
    "primary_metric_name": "F1 (>50K) difference (SMOTE - no resampling)",
    "primary_metric_value": round(float(diff), 4),
    "direction": (
        "SMOTE changes F1 by >0.02" if abs(diff) > 0.02 else "SMOTE does not change F1 by >0.02"
    ),
    "methodological_choices": (
        "Rows with '?' (missing) values dropped rather than imputed (~7% of rows, "
        "concentrated in workclass/occupation/native-country). fnlwgt (a census "
        "sampling weight) dropped as a non-demographic artifact column. "
        "75/25 stratified train/test split, random_state=42. Categorical features "
        "one-hot encoded (handle_unknown='ignore'); numeric features passed through "
        "unscaled since RandomForest is scale-invariant. RandomForestClassifier() used "
        "with all default hyperparameters (no random_state set on the classifier itself, "
        "so run-to-run variance is possible) for both arms, differing only in whether "
        "SMOTE (imblearn, default k_neighbors=5, random_state=42) was applied to the "
        "encoded training set before fitting. Evaluation metric is F1 restricted to the "
        "positive/minority class (>50K) on the held-out, non-resampled test set. "
        "A single train/test split was used (no cross-validation or repeated runs), "
        "so the reported difference reflects one realization of RF's inherent stochasticity."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print()
print(json.dumps(result, indent=2))
