"""
H5: Does SMOTE oversampling change the minority-class (>50K) F1 score by more
than 0.02 compared to no resampling, holding the classifier fixed as a
default-hyperparameter RandomForestClassifier()?
"""

import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

# Basic cleanup: strip whitespace from string columns, treat '?' as missing
for col in df.select_dtypes(include="object").columns:
    df[col] = df[col].str.strip()
df = df.replace("?", np.nan)

# Drop rows with missing values (small fraction of workclass/occupation/native-country)
df = df.dropna().reset_index(drop=True)

target_col = "class"
y = (df[target_col] == ">50K").astype(int)  # 1 = minority class of interest
X = df.drop(columns=[target_col])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

# Ordinal-encode categoricals so RandomForest (and SMOTE, which needs numeric
# input) can consume them. Simple, information-preserving enough for a tree
# model, and avoids a huge sparse one-hot matrix.
encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_encoded = X.copy()
X_encoded[cat_cols] = encoder.fit_transform(X[cat_cols])

print("Class balance overall:")
print(y.value_counts(normalize=True))

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain class counts:", y_train.value_counts().to_dict())
print("Test class counts:", y_test.value_counts().to_dict())

# --- Model 1: no resampling, default RandomForestClassifier ---
rf_plain = RandomForestClassifier(random_state=RANDOM_STATE)
rf_plain.fit(X_train, y_train)
pred_plain = rf_plain.predict(X_test)
f1_plain = f1_score(y_test, pred_plain, pos_label=1)

print("\n--- No resampling ---")
print(classification_report(y_test, pred_plain, target_names=["<=50K", ">50K"]))

# --- Model 2: SMOTE oversampling on training data only, same default RF ---
smote = SMOTE(random_state=RANDOM_STATE)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("\nResampled train class counts:", pd.Series(y_train_res).value_counts().to_dict())

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_res, y_train_res)
pred_smote = rf_smote.predict(X_test)
f1_smote = f1_score(y_test, pred_smote, pos_label=1)

print("\n--- SMOTE resampling ---")
print(classification_report(y_test, pred_smote, target_names=["<=50K", ">50K"]))

diff = f1_smote - f1_plain

print("\n=== Summary ===")
print(f"F1 (>50K), no resampling : {f1_plain:.4f}")
print(f"F1 (>50K), SMOTE         : {f1_smote:.4f}")
print(f"Difference (SMOTE - none): {diff:.4f}")
print(f"Exceeds 0.02 threshold?  : {abs(diff) > 0.02}")

result = {
    "hypothesis_id": "H5",
    "summary": (
        f"Applying SMOTE oversampling to the training data changed the minority-class "
        f"(>50K) F1 score from {f1_plain:.4f} (no resampling) to {f1_smote:.4f} (SMOTE), "
        f"a difference of {diff:+.4f}, which is "
        f"{'more' if abs(diff) > 0.02 else 'not more'} than the 0.02 threshold. "
        f"{'SMOTE did not help and in fact slightly hurt' if diff < 0 else 'SMOTE modestly improved'} "
        f"F1 for a default RandomForestClassifier on this dataset."
    ),
    "primary_metric_name": "F1 (>50K) difference (SMOTE - no resampling)",
    "primary_metric_value": round(float(diff), 4),
    "direction": (
        "difference exceeds 0.02 threshold"
        if abs(diff) > 0.02
        else "difference does not exceed 0.02 threshold"
    ),
    "methodological_choices": (
        "Dropped rows with missing values (marked '?') rather than imputing (~7% of rows). "
        "Encoded all categorical features with OrdinalEncoder (label encoding) rather than "
        "one-hot encoding, since RandomForest handles ordinal-coded categoricals reasonably "
        "and it keeps dimensionality low and compatible with SMOTE's numeric requirement. "
        "Used a single stratified 75/25 train/test split (random_state=42) rather than "
        "cross-validation, for simplicity and speed. Target treated as binary with '>50K' as "
        "the positive/minority class. RandomForestClassifier used with all default "
        "hyperparameters (n_estimators=100, etc.) as specified by the hypothesis. SMOTE "
        "(imbalanced-learn, default k_neighbors=5) applied only to the training set after the "
        "split, never to the test set, to avoid leakage. F1 score computed on the held-out "
        "test set for the '>50K' class only (pos_label=1)."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
