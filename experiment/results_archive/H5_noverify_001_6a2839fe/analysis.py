"""
H5: Does SMOTE oversampling change the minority-class (>50K) F1 score by more
than 0.02 compared to no resampling, holding the classifier fixed as a
default-hyperparameter RandomForestClassifier()?

Methodology
-----------
- Load adult_income.csv (48842 rows, target = `class`).
- Missing values are encoded as NaN (originally '?') in workclass,
  occupation, native-country. These are treated as their own category
  ("Missing") rather than dropped/imputed, to preserve all rows and
  avoid leaking information via imputation.
- Categorical features one-hot encoded; numeric features passed through
  unscaled (RandomForest is scale-invariant).
- Target encoded as binary: >50K = 1 (minority, positive class), <=50K = 0.
- To get a robust estimate (not dependent on one arbitrary split), we run
  5 independent stratified train/test splits (80/20), each with a fresh
  RandomForestClassifier() fit twice: once on the raw training fold, once
  on a SMOTE-resampled version of the same training fold (test fold is
  always left untouched/raw, as SMOTE must never be applied to test data).
  All other settings identical between the two arms.
- Classifier: RandomForestClassifier() with library defaults (n_estimators
  =100, no class_weight, no max_depth cap), as specified by the hypothesis.
- SMOTE: imblearn.over_sampling.SMOTE() with default settings (k_neighbors=5),
  applied only to the training fold, only on the numeric+one-hot-encoded
  feature matrix (SMOTE requires numeric input).
- Metric: F1 score on the minority class (>50K), i.e. pos_label=1.
- We report the mean difference (SMOTE F1 - no-resampling F1) across the
  5 splits, plus the std, as the primary result.
"""

import json
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore")

RANDOM_SEEDS = [0, 1, 2, 3, 4]

df = pd.read_csv("adult_income.csv")

# Treat missing values (NaN, originally '?') as an explicit category.
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != "class"]
num_cols = [c for c in df.columns if c not in cat_cols + ["class"]]

for c in cat_cols:
    df[c] = df[c].fillna("Missing").astype(str)

y = (df["class"].str.strip() == ">50K").astype(int)
X = df[num_cols + cat_cols]

print("Class balance:", y.value_counts(normalize=True).to_dict())
print("Minority class (>50K) count:", y.sum(), "/", len(y))

preprocessor = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

no_resample_f1s = []
smote_f1s = []

for seed in RANDOM_SEEDS:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    pre = preprocessor.fit(X_train)
    X_train_enc = pre.transform(X_train)
    X_test_enc = pre.transform(X_test)

    # Arm 1: no resampling
    rf_plain = RandomForestClassifier(random_state=seed)
    rf_plain.fit(X_train_enc, y_train)
    pred_plain = rf_plain.predict(X_test_enc)
    f1_plain = f1_score(y_test, pred_plain, pos_label=1)
    no_resample_f1s.append(f1_plain)

    # Arm 2: SMOTE-resampled training data
    sm = SMOTE(random_state=seed)
    X_train_sm, y_train_sm = sm.fit_resample(X_train_enc, y_train)

    rf_smote = RandomForestClassifier(random_state=seed)
    rf_smote.fit(X_train_sm, y_train_sm)
    pred_smote = rf_smote.predict(X_test_enc)
    f1_smote = f1_score(y_test, pred_smote, pos_label=1)
    smote_f1s.append(f1_smote)

    print(
        f"seed={seed}: F1 no-resample={f1_plain:.4f}  F1 SMOTE={f1_smote:.4f}  "
        f"diff={f1_smote - f1_plain:+.4f}"
    )

no_resample_f1s = np.array(no_resample_f1s)
smote_f1s = np.array(smote_f1s)
diffs = smote_f1s - no_resample_f1s

mean_no_resample = no_resample_f1s.mean()
mean_smote = smote_f1s.mean()
mean_diff = diffs.mean()
std_diff = diffs.std(ddof=1)

print("\n--- Summary over 5 stratified splits ---")
print(f"Mean F1 (no resampling): {mean_no_resample:.4f}")
print(f"Mean F1 (SMOTE):         {mean_smote:.4f}")
print(f"Mean diff (SMOTE - no resample): {mean_diff:+.4f} (std={std_diff:.4f})")
print(f"|mean diff| > 0.02 ? {abs(mean_diff) > 0.02}")

result = {
    "hypothesis_id": "H5",
    "summary": (
        f"Across 5 stratified 80/20 splits, SMOTE oversampling changed the mean "
        f"minority-class (>50K) F1 for a default RandomForestClassifier from "
        f"{mean_no_resample:.4f} (no resampling) to {mean_smote:.4f} (with SMOTE), "
        f"a mean difference of {mean_diff:+.4f}, which is "
        f"{'more' if abs(mean_diff) > 0.02 else 'not more'} than the 0.02 threshold. "
        f"{'SMOTE meaningfully changes' if abs(mean_diff) > 0.02 else 'SMOTE does not meaningfully change'} "
        f"F1 for this fixed default RF classifier on this dataset."
    ),
    "primary_metric_name": "Mean F1 difference (SMOTE - no resampling), minority class >50K",
    "primary_metric_value": float(mean_diff),
    "direction": (
        f"|diff| {'>' if abs(mean_diff) > 0.02 else '<='} 0.02: "
        f"{'SMOTE changes F1' if abs(mean_diff) > 0.02 else 'SMOTE does not change F1 by more than 0.02'} "
        f"({'SMOTE lower' if mean_diff < 0 else 'SMOTE higher'})"
    ),
    "methodological_choices": (
        "Missing values ('?') in workclass/occupation/native-country kept as an explicit "
        "'Missing' category rather than dropped or imputed. Categorical features one-hot "
        "encoded (unknown categories ignored at test time); numeric features passed through "
        "unscaled since RandomForest is scale-invariant. Target binarized with >50K as the "
        "positive/minority class. Used 5 independent stratified 80/20 train/test splits "
        "(seeds 0-4) rather than a single split, to get a stable estimate of the mean F1 "
        "difference and reduce sensitivity to split randomness; reported the mean difference "
        "across splits as the primary metric. RandomForestClassifier() used with library "
        "defaults exactly as specified by the hypothesis (no class_weight, no tuning). "
        "SMOTE() used with default settings (k_neighbors=5), applied only to the training "
        "fold after encoding (never to the test fold, to avoid leakage). Evaluation metric: "
        "F1 score with pos_label=1 (>50K)."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
