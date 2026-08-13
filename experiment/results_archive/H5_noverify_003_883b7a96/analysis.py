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
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

# The dataset uses "?" as missing-value marker in several categorical columns
# (workclass, occupation, native-country). Treat as its own category rather
# than dropping rows, to keep the full sample and mirror common practice.
df = df.replace("?", "Missing")

# Target: 1 = >50K (minority / positive class of interest), 0 = <=50K
df["class"] = df["class"].astype(str).str.strip()
y = (df["class"] == ">50K").astype(int)
X = df.drop(columns=["class"])

print("Class balance:")
print(y.value_counts(normalize=True))

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = [c for c in X.columns if c not in categorical_cols]
print("Categorical columns:", categorical_cols)
print("Numeric columns:", numeric_cols)

# ---------------------------------------------------------------------------
# Train/test split (stratified, single 70/30 split with fixed seed)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# Preprocessing: one-hot encode categoricals, pass numeric through as-is.
# RandomForest doesn't need scaling, so numeric features are left untouched.
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="passthrough",
)

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

# ---------------------------------------------------------------------------
# Condition A: No resampling, default RandomForestClassifier()
# ---------------------------------------------------------------------------
rf_plain = RandomForestClassifier(random_state=RANDOM_STATE)
rf_plain.fit(X_train_enc, y_train)
pred_plain = rf_plain.predict(X_test_enc)
f1_plain = f1_score(y_test, pred_plain, pos_label=1)

print("\n--- No resampling ---")
print(classification_report(y_test, pred_plain, target_names=["<=50K", ">50K"]))
print("F1 (>50K):", f1_plain)

# ---------------------------------------------------------------------------
# Condition B: SMOTE oversampling applied to training data only,
# same default RandomForestClassifier() on the resampled training set.
# ---------------------------------------------------------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)

print("\nTraining class balance after SMOTE:")
print(pd.Series(y_train_sm).value_counts())

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test_enc)
f1_smote = f1_score(y_test, pred_smote, pos_label=1)

print("\n--- SMOTE resampling ---")
print(classification_report(y_test, pred_smote, target_names=["<=50K", ">50K"]))
print("F1 (>50K):", f1_smote)

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
diff = f1_smote - f1_plain
print(f"\nF1 (>50K) no-resampling : {f1_plain:.4f}")
print(f"F1 (>50K) SMOTE         : {f1_smote:.4f}")
print(f"Difference (SMOTE - none): {diff:.4f}")
print(f"Exceeds 0.02 threshold?  : {abs(diff) > 0.02}")

# ---------------------------------------------------------------------------
# Write results
# ---------------------------------------------------------------------------
exceeds = abs(diff) > 0.02
direction = (
    f"SMOTE {'increases' if diff > 0 else 'decreases'} minority-class F1 by "
    f"{abs(diff):.4f} ({'exceeds' if exceeds else 'does not exceed'} 0.02 threshold)"
)

result = {
    "hypothesis_id": "H5",
    "summary": (
        f"Applying SMOTE to the training data changed the >50K-class F1 score "
        f"from {f1_plain:.4f} (no resampling) to {f1_smote:.4f} (SMOTE), a "
        f"difference of {diff:+.4f}. This is "
        f"{'larger' if exceeds else 'smaller'} than the 0.02 threshold, so "
        f"SMOTE did {'' if exceeds else 'not '}meaningfully change minority-class F1 "
        f"for a default RandomForestClassifier on this dataset/split."
    ),
    "primary_metric_name": "F1 (>50K) difference (SMOTE - no resampling)",
    "primary_metric_value": round(diff, 4),
    "direction": direction,
    "methodological_choices": (
        "Single stratified 70/30 train/test split (random_state=42), no cross-validation. "
        "'?' missing values in categorical columns recoded as a 'Missing' category rather "
        "than dropped. Categorical features one-hot encoded (OneHotEncoder, unknown "
        "categories ignored at test time); numeric features passed through unscaled "
        "(tree-based model, scaling not required). Classifier: RandomForestClassifier() "
        "with all default hyperparameters (n_estimators=100) for both conditions, only "
        "random_state fixed for reproducibility. Imbalance handling compared: (A) none, "
        "trained directly on the encoded, imbalanced training set; (B) imblearn SMOTE "
        "(default k_neighbors=5, random_state=42) applied to the encoded training set only "
        "(test set left untouched/imbalanced, as is standard practice). Positive/minority "
        "class defined as '>50K'. Metric: binary F1 score on the held-out test set for the "
        "'>50K' class specifically (sklearn f1_score with pos_label=1)."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
