"""
H5: Does SMOTE oversampling change minority-class (>50K) F1 for a default
RandomForestClassifier, compared to no resampling?
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

df = pd.read_csv("adult_income.csv")

# Treat literal "?" as missing too, in case it wasn't parsed as NaN already.
df = df.replace("?", np.nan)

y = (df["class"].str.strip() == ">50K").astype(int)
X = df.drop(columns=["class"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=np.number).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            SimpleImputer(strategy="median"),
            num_cols,
        ),
        (
            "cat",
            Pipeline(
                steps=[
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
    ]
)

# Fit preprocessing once on train, transform train/test consistently for both
# the resampled and non-resampled runs (fair comparison, same feature space).
X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

results = {}

# --- No resampling ---
rf_plain = RandomForestClassifier(random_state=RANDOM_STATE)
rf_plain.fit(X_train_enc, y_train)
pred_plain = rf_plain.predict(X_test_enc)
f1_plain = f1_score(y_test, pred_plain, pos_label=1)
results["no_resampling"] = {
    "f1_minority": f1_plain,
    "report": classification_report(y_test, pred_plain, target_names=["<=50K", ">50K"], output_dict=True),
}

# --- SMOTE oversampling on training data only ---
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test_enc)
f1_smote = f1_score(y_test, pred_smote, pos_label=1)
results["smote"] = {
    "f1_minority": f1_smote,
    "report": classification_report(y_test, pred_smote, target_names=["<=50K", ">50K"], output_dict=True),
}

diff = f1_smote - f1_plain

print("Train class balance (before SMOTE):", np.bincount(y_train))
print("Train class balance (after SMOTE):", np.bincount(y_train_sm))
print()
print(f"F1 (>50K), no resampling: {f1_plain:.4f}")
print(f"F1 (>50K), SMOTE:         {f1_smote:.4f}")
print(f"Difference (SMOTE - no resampling): {diff:+.4f}")
print()
print("Exceeds |0.02| threshold:", abs(diff) > 0.02)

summary = (
    f"Applying SMOTE changed the >50K-class F1 score from {f1_plain:.4f} (no resampling) to "
    f"{f1_smote:.4f} (with SMOTE), a difference of {diff:+.4f}. "
    f"This {'exceeds' if abs(diff) > 0.02 else 'does not exceed'} the 0.02 threshold, so SMOTE "
    f"{'meaningfully changes' if abs(diff) > 0.02 else 'does not meaningfully change'} minority-class F1 "
    "for a default-hyperparameter random forest on this dataset."
)

result = {
    "hypothesis_id": "H5",
    "summary": summary,
    "primary_metric_name": "F1 (>50K) difference (SMOTE - no resampling)",
    "primary_metric_value": round(diff, 4),
    "direction": (
        f"|diff| {'>' if abs(diff) > 0.02 else '<='} 0.02: SMOTE "
        f"{'moves' if abs(diff) > 0.02 else 'does not move'} F1 by more than 0.02 "
        f"({'improves' if diff > 0 else 'reduces' if diff < 0 else 'no change to'} it)"
    ),
    "methodological_choices": (
        "75/25 stratified train/test split (random_state=42); literal '?' strings recoded as NaN "
        "alongside existing NaNs; numeric features median-imputed, categorical features "
        "most-frequent-imputed then one-hot encoded (unknown categories ignored at test time); "
        "preprocessing pipeline fit once on the training set and reused for both conditions so the "
        "only difference between arms is SMOTE; SMOTE (imbalanced-learn, random_state=42, default "
        "k_neighbors=5) applied only to the encoded training data, never to the test set; "
        "RandomForestClassifier used with library-default hyperparameters (n_estimators=100) in "
        "both arms, random_state=42; F1 computed for the >50K (minority, pos_label=1) class on the "
        "held-out test set; threshold for 'meaningful change' taken as |ΔF1| > 0.02 per the "
        "research question."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print()
print(json.dumps(result, indent=2))
