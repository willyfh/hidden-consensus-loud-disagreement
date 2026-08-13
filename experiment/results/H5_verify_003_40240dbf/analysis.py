"""
H5: Does SMOTE oversampling change the minority-class (>50K) F1 score by more than
0.02 compared to no resampling, holding the classifier fixed as a default
RandomForestClassifier()?
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

# Basic cleaning: treat '?' as missing category (keep as its own level - simplest,
# avoids dropping ~7% of rows), strip whitespace just in case.
for c in df.select_dtypes(include=["object", "str"]).columns:
    df[c] = df[c].str.strip()
    if df[c].isna().any():
        df[c] = df[c].fillna("Missing")

print("Shape:", df.shape)
print(df["class"].value_counts())
print(df.isna().sum().sum(), "NaNs")

target = "class"
y = (df[target] == ">50K").astype(int)
X = df.drop(columns=[target])

cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]
print("Categorical cols:", cat_cols)
print("Numeric cols:", num_cols)

# Encode categoricals with OrdinalEncoder (RF doesn't need one-hot; keeps SMOTE's
# k-NN interpolation in a lower-dimensional, denser space than one-hot would).
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_enc = X.copy()
X_enc[cat_cols] = enc.fit_transform(X[cat_cols])
X_enc = X_enc.astype(float)

# ---------------------------------------------------------------------------
# Primary analysis: single stratified train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

print("\nTrain class balance:", y_train.value_counts(normalize=True).to_dict())

# No resampling
rf_plain = RandomForestClassifier(random_state=RANDOM_STATE)
rf_plain.fit(X_train, y_train)
pred_plain = rf_plain.predict(X_test)
f1_plain = f1_score(y_test, pred_plain, pos_label=1)

# With SMOTE
sm = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test)
f1_smote = f1_score(y_test, pred_smote, pos_label=1)

diff = f1_smote - f1_plain
print(f"\nPrimary split: F1 no-resample = {f1_plain:.4f}, F1 SMOTE = {f1_smote:.4f}, diff = {diff:.4f}")

# ---------------------------------------------------------------------------
# Stability check: repeated stratified k-fold CV with different seeds
# ---------------------------------------------------------------------------
seeds = [0, 1, 2, 3, 4]
n_splits = 5

plain_f1s = []
smote_f1s = []
diffs = []

for seed in seeds:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X_enc, y):
        Xtr, Xte = X_enc.iloc[train_idx], X_enc.iloc[test_idx]
        ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

        rf1 = RandomForestClassifier(random_state=seed)
        rf1.fit(Xtr, ytr)
        f1_no = f1_score(yte, rf1.predict(Xte), pos_label=1)

        sm2 = SMOTE(random_state=seed)
        Xtr_sm, ytr_sm = sm2.fit_resample(Xtr, ytr)
        rf2 = RandomForestClassifier(random_state=seed)
        rf2.fit(Xtr_sm, ytr_sm)
        f1_sm = f1_score(yte, rf2.predict(Xte), pos_label=1)

        plain_f1s.append(f1_no)
        smote_f1s.append(f1_sm)
        diffs.append(f1_sm - f1_no)

plain_f1s = np.array(plain_f1s)
smote_f1s = np.array(smote_f1s)
diffs = np.array(diffs)

print(f"\nRepeated CV ({len(seeds)}x{n_splits}-fold, {len(diffs)} folds total):")
print(f"  No-resample F1: mean={plain_f1s.mean():.4f}, sd={plain_f1s.std():.4f}")
print(f"  SMOTE F1:       mean={smote_f1s.mean():.4f}, sd={smote_f1s.std():.4f}")
print(f"  Diff (SMOTE-plain): mean={diffs.mean():.4f}, sd={diffs.std():.4f}, "
      f"min={diffs.min():.4f}, max={diffs.max():.4f}")
print(f"  95% CI on diff (normal approx): "
      f"[{diffs.mean() - 1.96*diffs.std()/np.sqrt(len(diffs)):.4f}, "
      f"{diffs.mean() + 1.96*diffs.std()/np.sqrt(len(diffs)):.4f}]")

exceeds_threshold = abs(diffs.mean()) > 0.02
print(f"\nMean |diff| > 0.02 threshold? {exceeds_threshold}")

import json

result = {
    "primary_split_f1_no_resample": float(f1_plain),
    "primary_split_f1_smote": float(f1_smote),
    "primary_split_diff": float(diff),
    "cv_diff_mean": float(diffs.mean()),
    "cv_diff_sd": float(diffs.std()),
    "cv_diff_min": float(diffs.min()),
    "cv_diff_max": float(diffs.max()),
}
with open("cv_debug_results.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nSaved debug results to cv_debug_results.json")
