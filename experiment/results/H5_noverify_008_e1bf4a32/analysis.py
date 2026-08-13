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
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

# The raw UCI/OpenML dump uses "?" for missing categorical values.
df = df.replace("?", np.nan)

# Target: >50K is the minority class of interest.
df["class"] = df["class"].astype(str).str.strip()
y = (df["class"] == ">50K").astype(int)
print("Class balance:\n", df["class"].value_counts(normalize=True))

X = df.drop(columns=["class"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# Simple missing-value handling for categoricals (small fraction of rows,
# mostly in workclass/occupation/native-country): treat "missing" as its own
# category rather than dropping rows, so we don't discard data.
for c in cat_cols:
    X[c] = X[c].fillna("Missing")

# ---------------------------------------------------------------------------
# Train/test split (stratified to preserve class ratio in the held-out test
# set, since the test set should reflect the true population distribution
# regardless of how the training data is resampled).
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough",
)

# Fit preprocessor once on training data, reuse encoded matrices for both
# the no-resampling and SMOTE runs so the only difference between the two
# conditions is the resampling step itself.
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

# ---------------------------------------------------------------------------
# Condition A: No resampling, default RandomForestClassifier()
# ---------------------------------------------------------------------------
rf_baseline = RandomForestClassifier(random_state=RANDOM_STATE)
rf_baseline.fit(X_train_enc, y_train)
pred_baseline = rf_baseline.predict(X_test_enc)
f1_baseline = f1_score(y_test, pred_baseline, pos_label=1)

# ---------------------------------------------------------------------------
# Condition B: SMOTE oversampling on training data, same default RF
# ---------------------------------------------------------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train_enc, y_train)

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_res, y_train_res)
pred_smote = rf_smote.predict(X_test_enc)
f1_smote = f1_score(y_test, pred_smote, pos_label=1)

diff = f1_smote - f1_baseline

print("\n--- No resampling ---")
print(classification_report(y_test, pred_baseline, target_names=["<=50K", ">50K"]))
print("F1 (>50K):", f1_baseline)

print("\n--- SMOTE ---")
print(classification_report(y_test, pred_smote, target_names=["<=50K", ">50K"]))
print("F1 (>50K):", f1_smote)

print(f"\nF1 difference (SMOTE - baseline): {diff:.4f}")
print(f"Training set size: baseline={X_train_enc.shape[0]}, SMOTE-resampled={X_train_res.shape[0]}")

exceeds_threshold = abs(diff) > 0.02

# ---------------------------------------------------------------------------
# Write result
# ---------------------------------------------------------------------------
result = {
    "hypothesis_id": "H5",
    "summary": (
        f"Applying SMOTE to the training data changed the >50K-class F1 score "
        f"from {f1_baseline:.4f} (no resampling) to {f1_smote:.4f} (SMOTE), a "
        f"difference of {diff:+.4f}, which "
        f"{'exceeds' if exceeds_threshold else 'does not exceed'} the 0.02 threshold. "
        f"{'SMOTE meaningfully changed' if exceeds_threshold else 'SMOTE did not meaningfully change'} "
        f"minority-class F1 for a default RandomForestClassifier on this dataset."
    ),
    "primary_metric_name": "F1 (>50K) difference (SMOTE - no resampling)",
    "primary_metric_value": round(diff, 4),
    "direction": (
        f"|diff| {'>' if exceeds_threshold else '<='} 0.02: "
        f"{'SMOTE changes F1' if exceeds_threshold else 'SMOTE ~ no resampling'} "
        f"(baseline F1={f1_baseline:.4f}, SMOTE F1={f1_smote:.4f})"
    ),
    "methodological_choices": (
        "75/25 stratified train/test split (random_state=42), test set left untouched "
        "(reflects true population imbalance) and only the training set resampled with SMOTE. "
        "'?' values in categorical columns (workclass, occupation, native-country) recoded as an "
        "explicit 'Missing' category rather than dropped, to retain all rows. Categorical features "
        "one-hot encoded (handle_unknown='ignore'); numeric features passed through unscaled since "
        "RandomForestClassifier is scale-invariant. Both RandomForestClassifier() and SMOTE() used "
        "with fully default hyperparameters except random_state=42 for reproducibility. Minority "
        "class defined as '>50K'. F1 computed via sklearn f1_score with pos_label=1 (>50K). "
        "Single train/test split used (no cross-validation) per the question's framing as a direct "
        "before/after comparison; a CV-based estimate would give a less noisy but more expensive answer."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
print(json.dumps(result, indent=2))
