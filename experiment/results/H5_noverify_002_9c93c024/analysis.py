"""
H5: Does SMOTE oversampling change minority-class (>50K) F1 score by more than
0.02 compared to no resampling, holding the classifier fixed as a
default-hyperparameter RandomForestClassifier()?
"""
import json

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

target_col = "class"
y = (df[target_col] == ">50K").astype(int)  # 1 = minority/positive class (>50K)
X = df.drop(columns=[target_col])

# some pandas versions read strings as 'str'/'string' dtype rather than 'object'
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)
print("Class balance overall:\n", y.value_counts(normalize=True))

# ---------------------------------------------------------------------------
# Train / test split (stratified, held out untouched by any resampling)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# Preprocessing: impute missing categoricals with a constant "missing"
# category, one-hot encode categoricals, pass numeric columns through.
# Missing values in this dataset only occur in workclass/occupation/
# native-country (encoded as NaN after read_csv, originally "?").
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
        ("num", SimpleImputer(strategy="median"), num_cols),
    ]
)

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

# Densify: SMOTE and RandomForest both work fine on dense arrays and the
# encoded feature count here is small enough (~100 cols) to not matter.
if hasattr(X_train_enc, "toarray"):
    X_train_enc = X_train_enc.toarray()
if hasattr(X_test_enc, "toarray"):
    X_test_enc = X_test_enc.toarray()

print("Encoded train shape:", X_train_enc.shape)

# ---------------------------------------------------------------------------
# Condition A: No resampling, default-hyperparameter RandomForestClassifier
# ---------------------------------------------------------------------------
rf_no_resample = RandomForestClassifier(random_state=RANDOM_STATE)
rf_no_resample.fit(X_train_enc, y_train)
pred_no_resample = rf_no_resample.predict(X_test_enc)
f1_no_resample = f1_score(y_test, pred_no_resample, pos_label=1)

print("\n=== No resampling ===")
print(classification_report(y_test, pred_no_resample, target_names=["<=50K", ">50K"]))
print("F1 (>50K):", f1_no_resample)

# ---------------------------------------------------------------------------
# Condition B: SMOTE oversampling applied to training data only, same
# default-hyperparameter RandomForestClassifier
# ---------------------------------------------------------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_smote, y_train_smote = smote.fit_resample(X_train_enc, y_train)

print("\nClass balance after SMOTE:\n", pd.Series(y_train_smote).value_counts())

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_smote, y_train_smote)
pred_smote = rf_smote.predict(X_test_enc)
f1_smote = f1_score(y_test, pred_smote, pos_label=1)

print("\n=== SMOTE oversampling ===")
print(classification_report(y_test, pred_smote, target_names=["<=50K", ">50K"]))
print("F1 (>50K):", f1_smote)

# ---------------------------------------------------------------------------
# Compare
# ---------------------------------------------------------------------------
diff = f1_smote - f1_no_resample
print(f"\nF1 (>50K) no-resample : {f1_no_resample:.4f}")
print(f"F1 (>50K) SMOTE       : {f1_smote:.4f}")
print(f"Difference (SMOTE - no-resample): {diff:.4f}")
print(f"|Difference| > 0.02 ? {abs(diff) > 0.02}")

# ---------------------------------------------------------------------------
# Write result.json
# ---------------------------------------------------------------------------
exceeds = abs(diff) > 0.02
direction = (
    f"SMOTE {'increases' if diff > 0 else 'decreases'} minority-class F1 by "
    f"{abs(diff):.4f} ({'exceeds' if exceeds else 'does not exceed'} the 0.02 threshold)"
)

result = {
    "hypothesis_id": "H5",
    "summary": (
        f"Applying SMOTE to the training data changed the >50K F1 score from "
        f"{f1_no_resample:.4f} (no resampling) to {f1_smote:.4f} (SMOTE), a difference of "
        f"{diff:+.4f}, which does {'exceed' if exceeds else 'not exceed'} the 0.02 threshold. "
        f"With a default RandomForestClassifier, SMOTE {'meaningfully changes' if exceeds else 'has little practical effect on'} minority-class F1 on this dataset."
    ),
    "primary_metric_name": "F1 (>50K) difference, SMOTE minus no-resampling",
    "primary_metric_value": round(diff, 4),
    "direction": direction,
    "methodological_choices": (
        "One 75/25 stratified train/test split (random_state=42), test set never resampled. "
        "Missing values in workclass/occupation/native-country (originally '?') imputed as a "
        "'missing' category for categoricals; no missingness in numeric columns. Categorical "
        "features one-hot encoded (handle_unknown='ignore'); numeric features passed through "
        "unscaled (tree ensembles are scale-invariant). Target binarized with '>50K' as the "
        "positive/minority class. Classifier: RandomForestClassifier() with all default "
        "hyperparameters (n_estimators=100 etc.), random_state=42 for reproducibility, fixed "
        "identically across both conditions. Resampling: imblearn SMOTE(random_state=42) with "
        "default k_neighbors=5, applied only to the training set after preprocessing/encoding. "
        "Metric: F1 score for the '>50K' class specifically (not macro/weighted-averaged), "
        "since the question asks about minority-class F1. Single train/test split used rather "
        "than cross-validation for simplicity; results could vary with a different split or with "
        "CV-averaged estimates."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
