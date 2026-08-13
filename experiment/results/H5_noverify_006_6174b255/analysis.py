"""
H5: Does SMOTE oversampling change minority-class (>50K) F1 for a default
RandomForestClassifier, compared to no resampling?
"""
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

# Missing values are encoded as literal "?" strings in this dataset variant,
# but here they already came through as NaN (checked during exploration).
# Treat NaNs as their own category rather than dropping rows, since
# workclass/occupation/native-country missingness is informative and
# dropping would lose ~2.4k rows.
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("class")
num_cols = [c for c in df.columns if c not in cat_cols + ["class"]]

for c in cat_cols:
    df[c] = df[c].fillna("Missing").astype(str)

X = df[cat_cols + num_cols].copy()
y = (df["class"].str.strip() == ">50K").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# Ordinal-encode categoricals fit on train only, applied to test.
# (Tree-based RF doesn't need one-hot; ordinal encoding keeps things simple
# and fast, and SMOTE can operate on the resulting numeric feature space.)
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_enc = X_train.copy()
X_test_enc = X_test.copy()
X_train_enc[cat_cols] = encoder.fit_transform(X_train[cat_cols])
X_test_enc[cat_cols] = encoder.transform(X_test[cat_cols])

print("Train class balance:", y_train.value_counts().to_dict())
print("Test class balance:", y_test.value_counts().to_dict())

# --- Baseline: no resampling, default RandomForestClassifier ---
rf_base = RandomForestClassifier(random_state=RANDOM_STATE)
rf_base.fit(X_train_enc, y_train)
pred_base = rf_base.predict(X_test_enc)
f1_base = f1_score(y_test, pred_base, pos_label=1)
print("\n--- No resampling ---")
print(classification_report(y_test, pred_base, target_names=["<=50K", ">50K"]))

# --- SMOTE oversampling on training data only, same classifier/params ---
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)
print("\nAfter SMOTE, train class balance:", pd.Series(y_train_sm).value_counts().to_dict())

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test_enc)
f1_smote = f1_score(y_test, pred_smote, pos_label=1)
print("\n--- SMOTE ---")
print(classification_report(y_test, pred_smote, target_names=["<=50K", ">50K"]))

diff = f1_smote - f1_base
print(f"\nF1 (>50K) no-resampling : {f1_base:.4f}")
print(f"F1 (>50K) SMOTE         : {f1_smote:.4f}")
print(f"Difference (SMOTE - baseline): {diff:.4f}")
print(f"|diff| > 0.02 ? {abs(diff) > 0.02}")

result = {
    "hypothesis_id": "H5",
    "summary": (
        f"Applying SMOTE oversampling to the training data changed the minority-class "
        f"(>50K) F1 score from {f1_base:.4f} (no resampling) to {f1_smote:.4f} (SMOTE), "
        f"a difference of {diff:+.4f}, which is "
        f"{'more' if abs(diff) > 0.02 else 'not more'} than the 0.02 threshold. "
        f"With a default RandomForestClassifier, SMOTE {'meaningfully changed' if abs(diff) > 0.02 else 'did not meaningfully change'} minority-class F1 on held-out test data."
    ),
    "primary_metric_name": "F1 (>50K) difference (SMOTE - no resampling)",
    "primary_metric_value": round(float(diff), 4),
    "direction": (
        f"SMOTE {'increases' if diff > 0 else 'decreases'} minority F1 by {abs(diff):.4f}"
        f" ({'>' if abs(diff) > 0.02 else '<='} 0.02 threshold)"
    ),
    "methodological_choices": (
        "75/25 stratified train/test split, random_state=42. Missing categorical values "
        "(workclass/occupation/native-country, coded as NaN) filled with a 'Missing' "
        "category rather than dropped. Categorical features ordinal-encoded (fit on train "
        "only) rather than one-hot, since RandomForest handles ordinal-coded splits fine "
        "and this avoids high dimensionality; SMOTE was applied to this same encoded "
        "numeric space, treating ordinal codes as continuous (a simplification also made "
        "for computational reasons — one-hot + SMOTE is more standard but more expensive). "
        "SMOTE (imblearn, default k_neighbors=5) applied to training data only, never to "
        "test data. RandomForestClassifier used with fully default hyperparameters in both "
        "conditions (only random_state fixed for reproducibility). Minority class = '>50K'. "
        "Metric: F1 score for the '>50K' class specifically (not macro/weighted), evaluated "
        "on a single held-out test split (no cross-validation / repeated resampling to "
        "estimate variance of the difference)."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
