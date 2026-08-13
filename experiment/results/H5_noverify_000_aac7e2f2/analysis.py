"""
H5: Does SMOTE oversampling change the minority-class (>50K) F1 score by more than
0.02 compared to no resampling, holding the classifier fixed as a default
RandomForestClassifier()?

Methodology
-----------
- Load adult_income.csv (48842 rows). Target = `class` (<=50K / >50K), positive
  class = '>50K' (the minority class, ~24% of rows).
- Missing values appear as NaN in `workclass`, `occupation`, `native-country`
  (originally '?' in the raw UCI data). These are treated as their own
  "Missing" category rather than being dropped, to avoid throwing away rows.
- Categorical features one-hot encoded (pd.get_dummies); numeric features left
  as-is (RandomForest does not require scaling).
- Single stratified train/test split (80/20, random_state=42) so both arms of
  the experiment (with/without SMOTE) are evaluated on the identical held-out
  test set -- SMOTE is fit only on the training data, never the test data.
- Classifier: RandomForestClassifier() with library defaults (n_estimators=100,
  random_state=42 for reproducibility only -- no other hyperparameter tuning),
  fixed identically across both arms per the research question.
- Arm A (baseline): fit RF directly on the raw (imbalanced) training data.
- Arm B (SMOTE): apply imblearn.over_sampling.SMOTE(random_state=42) to the
  training data only, then fit the same RF on the resampled data.
- Metric: F1 score of the minority class ('>50K') on the untouched test set,
  via sklearn.metrics.f1_score(pos_label='>50K').
- Decision rule: |F1_smote - F1_baseline| > 0.02 => SMOTE "changes" the
  minority F1 score meaningfully for this research question.
"""

import json

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
POS_LABEL = ">50K"

df = pd.read_csv("adult_income.csv")

cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != "class"]
for c in cat_cols:
    df[c] = df[c].fillna("Missing")

X = df.drop(columns=["class"])
y = df["class"]

X = pd.get_dummies(X, columns=cat_cols, drop_first=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("Train class distribution:")
print(y_train.value_counts())
print("\nTest class distribution:")
print(y_test.value_counts())

# --- Arm A: no resampling ---
rf_baseline = RandomForestClassifier(random_state=RANDOM_STATE)
rf_baseline.fit(X_train, y_train)
pred_baseline = rf_baseline.predict(X_test)

f1_baseline = f1_score(y_test, pred_baseline, pos_label=POS_LABEL)
prec_baseline = precision_score(y_test, pred_baseline, pos_label=POS_LABEL)
rec_baseline = recall_score(y_test, pred_baseline, pos_label=POS_LABEL)

# --- Arm B: SMOTE on training data only ---
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

print("\nSMOTE-resampled train class distribution:")
print(y_train_sm.value_counts())

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test)

f1_smote = f1_score(y_test, pred_smote, pos_label=POS_LABEL)
prec_smote = precision_score(y_test, pred_smote, pos_label=POS_LABEL)
rec_smote = recall_score(y_test, pred_smote, pos_label=POS_LABEL)

diff = f1_smote - f1_baseline

print("\n=== Results ===")
print(f"Baseline (no resampling): F1={f1_baseline:.4f} P={prec_baseline:.4f} R={rec_baseline:.4f}")
print(f"SMOTE:                    F1={f1_smote:.4f} P={prec_smote:.4f} R={rec_smote:.4f}")
print(f"Difference (SMOTE - baseline): {diff:.4f}")
print(f"Exceeds 0.02 threshold: {abs(diff) > 0.02}")

result = {
    "hypothesis_id": "H5",
    "summary": (
        f"Applying SMOTE to the training data changed the minority-class (>50K) F1 score "
        f"from {f1_baseline:.4f} (no resampling) to {f1_smote:.4f} (with SMOTE), a difference "
        f"of {diff:+.4f}, which does {'exceed' if abs(diff) > 0.02 else 'NOT exceed'} the 0.02 threshold. "
        f"SMOTE shifted the precision/recall balance (recall {'up' if rec_smote > rec_baseline else 'down'}, "
        f"precision {'up' if prec_smote > prec_baseline else 'down'}) without materially changing the overall F1."
        if abs(diff) <= 0.02 else
        f"Applying SMOTE to the training data changed the minority-class (>50K) F1 score "
        f"from {f1_baseline:.4f} (no resampling) to {f1_smote:.4f} (with SMOTE), a difference "
        f"of {diff:+.4f}, which exceeds the 0.02 threshold."
    ),
    "primary_metric_name": "F1 (>50K) difference (SMOTE - no resampling)",
    "primary_metric_value": round(diff, 4),
    "direction": (
        f"SMOTE {'improves' if diff > 0 else 'reduces'} minority F1 by {abs(diff):.4f}; "
        f"{'exceeds' if abs(diff) > 0.02 else 'below'} 0.02 threshold"
    ),
    "methodological_choices": (
        "Missing categorical values (workclass/occupation/native-country, originally '?') "
        "kept as an explicit 'Missing' category rather than dropped. Categorical features "
        "one-hot encoded via pd.get_dummies; numeric features used as-is (no scaling, "
        "unnecessary for tree-based models). Single stratified 80/20 train/test split "
        "(random_state=42); SMOTE fit only on training fold, test set left untouched and "
        "identical across both arms. RandomForestClassifier used with library defaults "
        "(only random_state=42 set, for reproducibility) per the research question's "
        "specification. SMOTE from imbalanced-learn with default k_neighbors=5 and "
        "random_state=42. Minority class F1 computed with pos_label='>50K'. A single "
        "train/test split (rather than cross-validation) was used for simplicity and "
        "because the dataset is large enough (48842 rows) for a stable held-out estimate; "
        "a different researcher might prefer repeated k-fold CV to average out split variance."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
