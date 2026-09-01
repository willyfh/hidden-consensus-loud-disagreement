"""
H2: Does RandomForestClassifier() beat LogisticRegression() (both sklearn defaults)
    on stratified 5-fold CV ROC-AUC for the Adult income dataset?

Design notes (my methodological choices):
  - Both models are wrapped in the SAME preprocessing pipeline so the only thing
    that differs is the estimator. Preprocessing is fit inside each CV fold.
  - Numeric: median impute + StandardScaler (scaling matters for lbfgs LogReg;
    it is harmless for RF). Categorical: most-frequent impute + one-hot.
  - All 14 features used, including fnlwgt (a survey sampling weight). Kept for
    both models equally; a sensitivity run drops it.
  - Metric: ROC-AUC on predict_proba, stratified 5-fold, shuffled, seed 0.
  - Paired (same folds) comparison -> per-fold differences.
Verification: 5x repeated stratified 5-fold (25 folds, seeds 0-4), paired
BCa-free percentile bootstrap over folds, plus an untouched 20% holdout re-test.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = 0


def load(drop_fnlwgt=False):
    df = pd.read_csv("adult_income.csv")
    df = df.replace("?", np.nan)
    y = (df["class"].str.strip() == ">50K").astype(int).to_numpy()
    X = df.drop(columns=["class"])
    if drop_fnlwgt:
        X = X.drop(columns=["fnlwgt"])
    return X, y


def make_pre(X):
    num = X.select_dtypes(include=np.number).columns.tolist()
    cat = X.select_dtypes(exclude=np.number).columns.tolist()
    return ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]
                ),
                num,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            ),
        ]
    )


def pipes(X, scale_numeric=True, logreg_kwargs=None):
    def pre():
        p = make_pre(X)
        if not scale_numeric:
            # drop the scaler to test the literal raw-default sensitivity
            p.transformers[0][1].set_params(sc="passthrough")
        return p

    lr = LogisticRegression(**(logreg_kwargs or {}))  # sklearn defaults
    rf = RandomForestClassifier(random_state=RNG, n_jobs=-1)  # sklearn defaults
    return (
        Pipeline([("pre", pre()), ("clf", lr)]),
        Pipeline([("pre", pre()), ("clf", rf)]),
    )


def run_cv(X, y, cv, **kw):
    lr_p, rf_p = pipes(X, **kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr = cross_val_score(lr_p, X, y, cv=cv, scoring="roc_auc")
        rf = cross_val_score(rf_p, X, y, cv=cv, scoring="roc_auc")
    return lr, rf


out = {}
X, y = load()
print(f"n={len(y)}  positives={y.mean():.4f}")

# ---------- primary analysis ----------
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
lr5, rf5 = run_cv(X, y, cv5)
d5 = rf5 - lr5
print("\n=== PRIMARY: stratified 5-fold, seed 0 ===")
print("LogReg per-fold:", np.round(lr5, 5), "mean", round(lr5.mean(), 5))
print("RF     per-fold:", np.round(rf5, 5), "mean", round(rf5.mean(), 5))
print("diff (RF-LR):", np.round(d5, 5), "mean", round(d5.mean(), 5))
out["primary"] = {
    "logreg_mean": lr5.mean(),
    "logreg_std": lr5.std(ddof=1),
    "rf_mean": rf5.mean(),
    "rf_std": rf5.std(ddof=1),
    "diff_mean": d5.mean(),
    "folds_rf_wins": int((d5 > 0).sum()),
}

# ---------- verification 1: 5x repeated stratified 5-fold ----------
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
lrR, rfR = run_cv(X, y, rcv)
dR = rfR - lrR
# paired t-based CI over the 25 folds (folds are correlated; treat as descriptive)
se = dR.std(ddof=1) / np.sqrt(len(dR))
ci = (dR.mean() - 1.96 * se, dR.mean() + 1.96 * se)
# percentile bootstrap over folds
bs = np.random.default_rng(RNG)
boot = np.array([bs.choice(dR, len(dR), replace=True).mean() for _ in range(10000)])
print("\n=== VERIFY 1: 5x repeated stratified 5-fold (25 folds, seeds 0-4) ===")
print(f"LogReg {lrR.mean():.5f} +/- {lrR.std(ddof=1):.5f}")
print(f"RF     {rfR.mean():.5f} +/- {rfR.std(ddof=1):.5f}")
print(f"diff   {dR.mean():.5f}  95% CI [{ci[0]:.5f}, {ci[1]:.5f}]")
print(f"bootstrap 95% CI [{np.percentile(boot,2.5):.5f}, {np.percentile(boot,97.5):.5f}]")
print(f"RF wins in {(dR>0).sum()}/{len(dR)} folds")
out["repeated_cv"] = {
    "logreg_mean": lrR.mean(),
    "rf_mean": rfR.mean(),
    "diff_mean": dR.mean(),
    "ci95_normal": list(ci),
    "ci95_bootstrap": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
    "folds_rf_wins": int((dR > 0).sum()),
    "n_folds": int(len(dR)),
}

# ---------- verification 2: untouched holdout re-test ----------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
lr_p, rf_p = pipes(X)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    lr_p.fit(Xtr, ytr)
    rf_p.fit(Xtr, ytr)
h_lr = roc_auc_score(yte, lr_p.predict_proba(Xte)[:, 1])
h_rf = roc_auc_score(yte, rf_p.predict_proba(Xte)[:, 1])
print("\n=== VERIFY 2: 80/20 holdout (seed 123) ===")
print(f"LogReg {h_lr:.5f}   RF {h_rf:.5f}   diff {h_rf-h_lr:+.5f}")
out["holdout"] = {"logreg": h_lr, "rf": h_rf, "diff": h_rf - h_lr}

# ---------- sensitivity: unscaled numerics (literal sklearn-default LogReg) ----------
lrU, rfU = run_cv(X, y, cv5, scale_numeric=False)
print("\n=== SENSITIVITY: no numeric scaling ===")
print(f"LogReg {lrU.mean():.5f}   RF {rfU.mean():.5f}   diff {(rfU-lrU).mean():+.5f}")
out["sens_unscaled"] = {
    "logreg_mean": lrU.mean(),
    "rf_mean": rfU.mean(),
    "diff_mean": (rfU - lrU).mean(),
}

# ---------- sensitivity: LogReg allowed to converge (max_iter=2000) ----------
lrC, rfC = run_cv(X, y, cv5, logreg_kwargs={"max_iter": 2000})
print("\n=== SENSITIVITY: LogReg max_iter=2000 ===")
print(f"LogReg {lrC.mean():.5f}   RF {rfC.mean():.5f}   diff {(rfC-lrC).mean():+.5f}")
out["sens_maxiter"] = {
    "logreg_mean": lrC.mean(),
    "rf_mean": rfC.mean(),
    "diff_mean": (rfC - lrC).mean(),
}

# ---------- sensitivity: drop fnlwgt ----------
X2, y2 = load(drop_fnlwgt=True)
lrF, rfF = run_cv(X2, y2, cv5)
print("\n=== SENSITIVITY: fnlwgt dropped ===")
print(f"LogReg {lrF.mean():.5f}   RF {rfF.mean():.5f}   diff {(rfF-lrF).mean():+.5f}")
out["sens_no_fnlwgt"] = {
    "logreg_mean": lrF.mean(),
    "rf_mean": rfF.mean(),
    "diff_mean": (rfF - lrF).mean(),
}

# ---------- sensitivity: unscaled AND converged (isolates scaling from convergence) ----------
lrUC, rfUC = run_cv(X, y, cv5, scale_numeric=False, logreg_kwargs={"max_iter": 1000})
print("\n=== SENSITIVITY: no scaling + max_iter=1000 ===")
print(f"LogReg {lrUC.mean():.5f}   RF {rfUC.mean():.5f}   diff {(rfUC-lrUC).mean():+.5f}")
out["sens_unscaled_converged"] = {
    "logreg_mean": lrUC.mean(),
    "rf_mean": rfUC.mean(),
    "diff_mean": (rfUC - lrUC).mean(),
}

with open("cv_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("\nwrote cv_results.json")
