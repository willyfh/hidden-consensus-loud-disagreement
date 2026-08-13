"""
H6: Is the model well-calibrated?

Approach:
1. Load adult_income.csv, clean missing values.
2. Encode categoricals, split into train/test (stratified).
3. Fit a gradient-boosted tree classifier (HistGradientBoostingClassifier) as the
   primary model, predicting P(class == '>50K').
4. Assess calibration on the held-out test set via:
   - Reliability diagram (10 equal-width bins of predicted probability)
   - Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
   - Brier score
   - Calibration intercept/slope from logistic regression of outcome on logit(p)
5. Compare raw model calibration to calibration after applying isotonic
   regression / Platt scaling (fit on a separate calibration split), to see
   how much recalibration would improve things and to contextualize how
   miscalibrated the raw model is.
"""

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score

RNG = 42

# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

cat_cols = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]
num_cols = ["age", "fnlwgt", "education-num", "capital-gain",
            "capital-loss", "hours-per-week"]

# Missing values are all in categorical columns (workclass, occupation,
# native-country) -- treat "missing" as its own category rather than
# dropping rows, since it may itself be informative (e.g. never worked).
for c in cat_cols:
    df[c] = df[c].fillna("Missing")

df["target"] = (df["class"].str.strip() == ">50K").astype(int)

X = df[cat_cols + num_cols].copy()
y = df["target"].values

enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X[cat_cols] = enc.fit_transform(X[cat_cols])

# ---------------------------------------------------------------------------
# 2. Split: train / calibration / test (60/20/20), stratified on target
# ---------------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=RNG, stratify=y
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RNG, stratify=y_temp
)

print(f"train={len(y_train)}  cal={len(y_cal)}  test={len(y_test)}")

# ---------------------------------------------------------------------------
# 3. Fit primary model (HistGradientBoosting) on train
# ---------------------------------------------------------------------------
categorical_mask = [c in cat_cols for c in X.columns]

model = HistGradientBoostingClassifier(
    random_state=RNG,
    categorical_features=categorical_mask,
    max_iter=200,
)
model.fit(X_train, y_train)

p_raw = model.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, p_raw)
print(f"Test ROC-AUC (raw model): {auc:.4f}")

# ---------------------------------------------------------------------------
# 4. Calibration diagnostics
# ---------------------------------------------------------------------------
def reliability_table(y_true, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(p, bins[1:-1], right=True)
    rows = []
    for b in range(n_bins):
        mask = bin_ids == b
        n = mask.sum()
        if n == 0:
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        rows.append({
            "bin": b,
            "bin_range": f"[{bins[b]:.1f}, {bins[b+1]:.1f})",
            "n": int(n),
            "mean_predicted": float(conf),
            "observed_freq": float(acc),
            "gap": float(conf - acc),
        })
    return rows


def ece_mce(y_true, p, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(p, bins[1:-1], right=True)
    N = len(p)
    ece = 0.0
    mce = 0.0
    for b in range(n_bins):
        mask = bin_ids == b
        n = mask.sum()
        if n == 0:
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        gap = abs(conf - acc)
        ece += (n / N) * gap
        mce = max(mce, gap)
    return ece, mce


table_raw = reliability_table(y_test, p_raw)
ece_raw, mce_raw = ece_mce(y_test, p_raw)
brier_raw = brier_score_loss(y_test, p_raw)

print("\nReliability table (raw HGB model):")
for r in table_raw:
    print(r)
print(f"ECE={ece_raw:.4f}  MCE={mce_raw:.4f}  Brier={brier_raw:.4f}")

# Calibration intercept & slope: logistic regression of y on logit(p)
eps = 1e-6
p_clip = np.clip(p_raw, eps, 1 - eps)
logit_p = np.log(p_clip / (1 - p_clip))
cal_lr = LogisticRegression()
cal_lr.fit(logit_p.reshape(-1, 1), y_test)
cal_slope = cal_lr.coef_[0][0]
cal_intercept = cal_lr.intercept_[0]
print(f"Calibration slope={cal_slope:.4f} (1=ideal), intercept={cal_intercept:.4f} (0=ideal)")

# ---------------------------------------------------------------------------
# 5. Recalibrate (isotonic, fit on separate cal split) and re-check
# ---------------------------------------------------------------------------
p_cal_raw = model.predict_proba(X_cal)[:, 1]
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_cal_raw, y_cal)
p_isotonic = iso.transform(p_raw)

table_iso = reliability_table(y_test, p_isotonic)
ece_iso, mce_iso = ece_mce(y_test, p_isotonic)
brier_iso = brier_score_loss(y_test, p_isotonic)
auc_iso = roc_auc_score(y_test, p_isotonic)

print("\nReliability table (isotonic-calibrated model):")
for r in table_iso:
    print(r)
print(f"ECE={ece_iso:.4f}  MCE={mce_iso:.4f}  Brier={brier_iso:.4f}  AUC={auc_iso:.4f}")

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
results = {
    "n_train": len(y_train),
    "n_cal": len(y_cal),
    "n_test": len(y_test),
    "raw_model": {
        "auc": auc,
        "brier": brier_raw,
        "ece": ece_raw,
        "mce": mce_raw,
        "calibration_slope": cal_slope,
        "calibration_intercept": cal_intercept,
        "reliability_table": table_raw,
    },
    "isotonic_calibrated_model": {
        "auc": auc_iso,
        "brier": brier_iso,
        "ece": ece_iso,
        "mce": mce_iso,
        "reliability_table": table_iso,
    },
}

with open("calibration_diagnostics.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nSaved diagnostics to calibration_diagnostics.json")
