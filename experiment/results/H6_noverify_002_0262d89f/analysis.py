"""
H6: Is the model well-calibrated?

We train a standard classifier (logistic regression, plus a gradient-boosted
tree for comparison) to predict `class` (<=50K vs >50K) on the Adult Income
dataset, then assess calibration of predicted probabilities on a held-out
test set using:
  - Reliability diagrams (binned observed frequency vs mean predicted prob)
  - Expected Calibration Error (ECE), 10 equal-width bins
  - Maximum Calibration Error (MCE)
  - Brier score
  - A comparison of raw vs a Platt/isotonic-recalibrated model, to see
    whether recalibration meaningfully improves things (evidence of whether
    the raw model was miscalibrated).
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import brier_score_loss, roc_auc_score

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")
df.columns = [c.strip() for c in df.columns]

# Clean: strip whitespace from string columns, treat '?' as missing category (keep as own level)
for c in df.select_dtypes(include=["object", "str"]).columns:
    df[c] = df[c].astype(str).str.strip()

df["target"] = (df["class"].str.replace(".", "", regex=False) == ">50K").astype(int)
print("Class balance:\n", df["target"].value_counts(normalize=True))

y = df["target"].values
X = df.drop(columns=["class", "target"])

cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "str"]).columns.tolist()
print("Categorical:", cat_cols)
print("Numeric:", num_cols)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
)

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# --- Model 1: Logistic Regression (well-known to produce fairly calibrated probs) ---
logreg = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
])
logreg.fit(X_train, y_train)
p_logreg = logreg.predict_proba(X_test)[:, 1]

# --- Model 2: Gradient Boosting (often less calibrated out-of-the-box, esp. with many trees) ---
gb = Pipeline([
    ("prep", preprocess),
    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=300, max_depth=3, learning_rate=0.1)),
])
gb.fit(X_train, y_train)
p_gb = gb.predict_proba(X_test)[:, 1]


def calibration_metrics(y_true, p_pred, n_bins=10):
    """Equal-width binning ECE/MCE + per-bin data for reliability diagram."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(p_pred, bins[1:-1], right=True)
    ece = 0.0
    mce = 0.0
    rows = []
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_ids == b
        count = mask.sum()
        if count == 0:
            continue
        conf = p_pred[mask].mean()
        acc = y_true[mask].mean()
        gap = abs(acc - conf)
        ece += (count / n) * gap
        mce = max(mce, gap)
        rows.append({
            "bin_low": bins[b], "bin_high": bins[b + 1],
            "count": int(count), "mean_predicted": conf, "observed_freq": acc, "gap": gap
        })
    return ece, mce, pd.DataFrame(rows)


ece_logreg, mce_logreg, tbl_logreg = calibration_metrics(y_test, p_logreg)
ece_gb, mce_gb, tbl_gb = calibration_metrics(y_test, p_gb)

brier_logreg = brier_score_loss(y_test, p_logreg)
brier_gb = brier_score_loss(y_test, p_gb)

auc_logreg = roc_auc_score(y_test, p_logreg)
auc_gb = roc_auc_score(y_test, p_gb)

print("\n=== Logistic Regression ===")
print(f"AUC: {auc_logreg:.4f}  Brier: {brier_logreg:.4f}  ECE: {ece_logreg:.4f}  MCE: {mce_logreg:.4f}")
print(tbl_logreg.to_string(index=False))

print("\n=== Gradient Boosting ===")
print(f"AUC: {auc_gb:.4f}  Brier: {brier_gb:.4f}  ECE: {ece_gb:.4f}  MCE: {mce_gb:.4f}")
print(tbl_gb.to_string(index=False))

# --- Recalibration check: does isotonic recalibration change the GB model appreciably? ---
# Use a fresh split so calibration fitting doesn't leak into the test set we already scored.
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_train, y_train, test_size=0.3, random_state=RANDOM_STATE, stratify=y_train
)

gb_fit = Pipeline([
    ("prep", preprocess),
    ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE, n_estimators=300, max_depth=3, learning_rate=0.1)),
])
gb_fit.fit(X_fit, y_fit)

gb_calibrated = CalibratedClassifierCV(FrozenEstimator(gb_fit), method="isotonic")
gb_calibrated.fit(X_cal, y_cal)
p_gb_calibrated = gb_calibrated.predict_proba(X_test)[:, 1]

ece_gb_cal, mce_gb_cal, tbl_gb_cal = calibration_metrics(y_test, p_gb_calibrated)
brier_gb_cal = brier_score_loss(y_test, p_gb_calibrated)

print("\n=== Gradient Boosting + Isotonic Recalibration ===")
print(f"Brier: {brier_gb_cal:.4f}  ECE: {ece_gb_cal:.4f}  MCE: {mce_gb_cal:.4f}")
print(tbl_gb_cal.to_string(index=False))

print("\n=== Summary ===")
print(f"LogReg  ECE={ece_logreg:.4f} MCE={mce_logreg:.4f} Brier={brier_logreg:.4f}")
print(f"GB      ECE={ece_gb:.4f} MCE={mce_gb:.4f} Brier={brier_gb:.4f}")
print(f"GB+Iso  ECE={ece_gb_cal:.4f} MCE={mce_gb_cal:.4f} Brier={brier_gb_cal:.4f}")

results = {
    "n_test": int(len(y_test)),
    "logreg": {"auc": auc_logreg, "brier": brier_logreg, "ece": ece_logreg, "mce": mce_logreg},
    "gb": {"auc": auc_gb, "brier": brier_gb, "ece": ece_gb, "mce": mce_gb},
    "gb_isotonic": {"brier": brier_gb_cal, "ece": ece_gb_cal, "mce": mce_gb_cal},
}
import json
with open("calibration_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved calibration_results.json")
