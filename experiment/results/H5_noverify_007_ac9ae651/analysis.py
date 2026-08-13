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
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

# The dataset uses "?" (with surrounding whitespace already stripped by pandas
# in this file) as the missing-value marker in workclass/occupation/native-country.
df = df.replace("?", np.nan)

print("Shape:", df.shape)
print("\nClass balance:\n", df["class"].value_counts(normalize=True))
print("\nMissing values:\n", df.isna().sum()[df.isna().sum() > 0])

# Target encoding
y = (df["class"].str.strip() == ">50K").astype(int)
X = df.drop(columns=["class"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
print("\nCategorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# Simple, consistent preprocessing: impute missing categoricals with a
# dedicated "Missing" category, then ordinal-encode all categoricals so a
# plain RandomForestClassifier() (no pipeline complexity) can consume them.
# Ordinal (integer-code) encoding is chosen over one-hot for simplicity and
# because tree ensembles handle arbitrary integer splits fine; this is a
# methodological choice another researcher might make differently.
X[cat_cols] = X[cat_cols].fillna("Missing")

encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

X_train = X_train_raw.copy()
X_test = X_test_raw.copy()
X_train[cat_cols] = encoder.fit_transform(X_train_raw[cat_cols])
X_test[cat_cols] = encoder.transform(X_test_raw[cat_cols])

print("\nTrain size:", X_train.shape, "Test size:", X_test.shape)
print("Train class balance:", y_train.value_counts(normalize=True).to_dict())

# --- Model 1: No resampling, default RandomForestClassifier ---
rf_plain = RandomForestClassifier(random_state=RANDOM_STATE)
rf_plain.fit(X_train, y_train)
pred_plain = rf_plain.predict(X_test)

f1_plain = f1_score(y_test, pred_plain, pos_label=1)
prec_plain = precision_score(y_test, pred_plain, pos_label=1)
rec_plain = recall_score(y_test, pred_plain, pos_label=1)

# --- Model 2: SMOTE oversampling on training data, then same default RF ---
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE, train class balance:", y_train_sm.value_counts(normalize=True).to_dict())

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test)

f1_smote = f1_score(y_test, pred_smote, pos_label=1)
prec_smote = precision_score(y_test, pred_smote, pos_label=1)
rec_smote = recall_score(y_test, pred_smote, pos_label=1)

diff = f1_smote - f1_plain

print("\n=== Results (minority class '>50K' = 1) ===")
print(f"No resampling : F1={f1_plain:.4f}  Precision={prec_plain:.4f}  Recall={rec_plain:.4f}")
print(f"SMOTE         : F1={f1_smote:.4f}  Precision={prec_smote:.4f}  Recall={rec_smote:.4f}")
print(f"Difference (SMOTE - no resampling): {diff:.4f}")
print(f"|diff| > 0.02 ? {abs(diff) > 0.02}")

result = {
    "hypothesis_id": "H5",
    "summary": (
        f"Applying SMOTE oversampling to the training data changed the minority-class "
        f"(>50K) F1 score from {f1_plain:.4f} (no resampling) to {f1_smote:.4f} (SMOTE), "
        f"a difference of {diff:+.4f}, which is "
        f"{'more' if abs(diff) > 0.02 else 'not more'} than the 0.02 threshold. "
        f"SMOTE {'increased' if diff > 0 else 'decreased'} recall "
        f"({rec_plain:.4f} -> {rec_smote:.4f}) while "
        f"{'increasing' if prec_smote > prec_plain else 'decreasing'} precision "
        f"({prec_plain:.4f} -> {prec_smote:.4f})."
    ),
    "primary_metric_name": "F1 difference (SMOTE - no resampling), minority class '>50K'",
    "primary_metric_value": round(diff, 4),
    "direction": (
        f"SMOTE {'meaningfully changes' if abs(diff) > 0.02 else 'does not meaningfully change'} "
        f"F1 ({'improves' if diff > 0 else 'worsens'} it by {abs(diff):.4f})"
    ),
    "methodological_choices": (
        "Missing values ('?') in workclass/occupation/native-country imputed with a "
        "dedicated 'Missing' category rather than dropped or mode-imputed, to preserve "
        "all 48842 rows. All categorical features ordinal-encoded (integer codes) rather "
        "than one-hot encoded, since RandomForestClassifier() handles arbitrary integer "
        "splits and this keeps feature count low. Single stratified 75/25 train/test split "
        "(random_state=42) rather than k-fold CV, for computational simplicity and because "
        "the dataset is large enough for a stable holdout estimate. SMOTE (imblearn, "
        "default k_neighbors=5) applied only to the training set (test set left at its "
        "natural ~24%/76% imbalance) to avoid data leakage. RandomForestClassifier used "
        "with library defaults (n_estimators=100, no class_weight, no max_depth limit) "
        "in both arms except random_state=42 for reproducibility, as specified by the "
        "research question. F1, precision, and recall computed with '>50K' as the "
        "positive/minority class."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
