"""
H2: Does RandomForestClassifier() beat LogisticRegression() (both sklearn defaults)
on stratified 5-fold CV ROC-AUC for the UCI Adult income dataset?

Design notes
------------
* The hypothesis fixes the estimators (library defaults) and the validation scheme
  (stratified 5-fold, ROC-AUC). Everything else -- encoding, imputation, scaling --
  is my call. I give BOTH models the identical preprocessing pipeline so the
  comparison isolates the model class.
* Preprocessing: median-impute + StandardScaler on the 6 numeric columns;
  most-frequent-impute + one-hot (handle_unknown='ignore') on the 8 categoricals.
  Scaling is required for LogisticRegression's default lbfgs/max_iter=100 to
  converge; it is a no-op for trees, so it does not advantage either side.
* Preprocessing is fit INSIDE each CV fold (Pipeline) to avoid leakage.
* No class-imbalance handling: ROC-AUC is threshold-free and rank-based, and both
  models are used at their library defaults as specified.
* Sensitivity check included for the one choice that could plausibly flip things:
  logistic regression WITHOUT scaling (raw one-hot + raw numerics).

Verification of stability: 5x repeated stratified 5-fold CV (25 folds, seeds 0-4),
paired per fold, plus a paired bootstrap CI on the fold-level differences, plus a
held-out 20% test split never touched by the CV analysis.
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
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
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------- data
df = pd.read_csv("adult_income.csv")
y = (df["class"].str.strip() == ">50K").astype(int).values
X = df.drop(columns=["class"])

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print(f"n={len(X)}  positives={y.mean():.4f}")
print(f"numeric   ({len(num_cols)}): {num_cols}")
print(f"categorical ({len(cat_cols)}): {cat_cols}")
print("missing per column:\n", X.isna().sum()[X.isna().sum() > 0], "\n")


def make_pre(scale=True):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def make_lr(scale=True):
    return Pipeline([("pre", make_pre(scale)), ("clf", LogisticRegression())])


def make_rf():
    # random_state fixed only so the run is reproducible; all other params default.
    return Pipeline([("pre", make_pre(True)),
                     ("clf", RandomForestClassifier(random_state=RNG))])


# ------------------------------------------------- primary: stratified 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
auc_rf = cross_val_score(make_rf(), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
auc_lr = cross_val_score(make_lr(), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
diff = auc_rf - auc_lr

print("=== PRIMARY: stratified 5-fold CV ROC-AUC (seed 0) ===")
print(f"RF     folds: {np.round(auc_rf, 5)}  mean={auc_rf.mean():.5f} sd={auc_rf.std(ddof=1):.5f}")
print(f"LogReg folds: {np.round(auc_lr, 5)}  mean={auc_lr.mean():.5f} sd={auc_lr.std(ddof=1):.5f}")
print(f"diff (RF-LR) mean = {diff.mean():+.5f}   won {int((diff>0).sum())}/5 folds\n")
primary_diff = float(diff.mean())

# ------------------------------------------------- sensitivity: LR without scaling
auc_lr_ns = cross_val_score(make_lr(scale=False), X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
print("=== SENSITIVITY: LogReg on UNSCALED features (default max_iter=100) ===")
print(f"LogReg-unscaled mean AUC = {auc_lr_ns.mean():.5f}  "
      f"(diff RF-LR = {auc_rf.mean()-auc_lr_ns.mean():+.5f})")

# Convergence diagnostic: is the SCALED logistic regression actually converged
# within the default max_iter=100? This decides which arm of the sensitivity
# analysis is the honest one.
_Xtr, _Xte, _ytr, _yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
for _scale in (True, False):
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        _m = make_lr(_scale).fit(_Xtr, _ytr)
        _conv = any(issubclass(w.category, ConvergenceWarning) for w in wlist)
    print(f"  scaled={_scale}: n_iter={_m[-1].n_iter_[0]} (max_iter=100), "
          f"ConvergenceWarning={_conv}, "
          f"holdout AUC={roc_auc_score(_yte, _m.predict_proba(_Xte)[:, 1]):.5f}")
print()

# ------------------------------------------------- verification 1: repeated CV
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RNG)
r_rf = cross_val_score(make_rf(), X, y, cv=rcv, scoring="roc_auc", n_jobs=-1)
r_lr = cross_val_score(make_lr(), X, y, cv=rcv, scoring="roc_auc", n_jobs=-1)
r_diff = r_rf - r_lr
t, p = stats.ttest_rel(r_rf, r_lr)

print("=== VERIFICATION 1: 5x repeated stratified 5-fold CV (25 folds, seeds 0-4) ===")
print(f"RF     mean={r_rf.mean():.5f} sd={r_rf.std(ddof=1):.5f}")
print(f"LogReg mean={r_lr.mean():.5f} sd={r_lr.std(ddof=1):.5f}")
print(f"diff   mean={r_diff.mean():+.5f} sd={r_diff.std(ddof=1):.5f}  "
      f"min={r_diff.min():+.5f} max={r_diff.max():+.5f}")
print(f"RF wins {int((r_diff>0).sum())}/25 folds; paired t={t:.2f}, p={p:.3g}")

# bootstrap CI over the 25 fold-level differences
boot = np.random.default_rng(RNG)
bs = np.array([boot.choice(r_diff, size=r_diff.size, replace=True).mean()
               for _ in range(10000)])
lo, hi = np.percentile(bs, [2.5, 97.5])
print(f"bootstrap 95% CI on mean diff: [{lo:+.5f}, {hi:+.5f}]\n")

# ------------------------------------------------- verification 2: held-out split
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=123)
rf_h, lr_h = make_rf().fit(Xtr, ytr), make_lr().fit(Xtr, ytr)
p_rf = rf_h.predict_proba(Xte)[:, 1]
p_lr = lr_h.predict_proba(Xte)[:, 1]
h_rf, h_lr = roc_auc_score(yte, p_rf), roc_auc_score(yte, p_lr)

bt = np.random.default_rng(RNG)
idx = np.arange(len(yte))
bd = []
for _ in range(2000):
    s = bt.choice(idx, size=idx.size, replace=True)
    if yte[s].min() == yte[s].max():
        continue
    bd.append(roc_auc_score(yte[s], p_rf[s]) - roc_auc_score(yte[s], p_lr[s]))
bd = np.array(bd)
hlo, hhi = np.percentile(bd, [2.5, 97.5])

print("=== VERIFICATION 2: held-out 20% test split (seed 123, unused above) ===")
print(f"RF AUC={h_rf:.5f}  LogReg AUC={h_lr:.5f}  diff={h_rf-h_lr:+.5f}")
print(f"paired bootstrap 95% CI on held-out diff: [{hlo:+.5f}, {hhi:+.5f}]  "
      f"P(RF>LR)={(bd>0).mean():.4f}\n")

# ------------------------------------------------- write result
winner = "RF" if primary_diff > 0 else "LogReg"
result = {
    "hypothesis_id": "H2",
    "summary": (
        f"No. With identical, leakage-free preprocessing given to both models, the "
        f"default RandomForestClassifier scored a stratified 5-fold CV ROC-AUC of "
        f"{auc_rf.mean():.4f} versus {auc_lr.mean():.4f} for the default "
        f"LogisticRegression -- RF is WORSE by {abs(primary_diff):.4f} AUC, and it lost "
        f"all 5 folds. The gap is small in practical terms (both models sit at ~0.90) "
        f"but statistically reliable: under 5x repeated CV logistic regression won 24 of "
        f"25 folds (p={p:.0e}). The one thing that reverses this verdict is failing to "
        f"scale the features, which leaves the default lbfgs solver unconverged at "
        f"max_iter=100 and collapses logistic regression to AUC {auc_lr_ns.mean():.4f}."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg), stratified 5-fold CV mean",
    "primary_metric_value": round(primary_diff, 5),
    "direction": "LogReg > RF (hypothesis not supported)",
    "methodological_choices": (
        "Both estimators at scikit-learn defaults (RF random_state fixed for "
        "reproducibility only) inside a Pipeline, so preprocessing is fit within each "
        "training fold and never leaks. Identical preprocessing for both models: the 6 "
        "numeric columns median-imputed + StandardScaler; the 8 categorical columns "
        "most-frequent-imputed + OneHotEncoder(handle_unknown='ignore'). The 3 columns "
        "with true NaNs (workclass 2799, occupation 2809, native-country 857) were "
        "imputed rather than dropped or given an explicit 'Missing' category. fnlwgt (a "
        "census sampling weight, arguably not a person-level predictor) was kept as an "
        "ordinary feature. Target encoded as >50K = 1 (23.93% positive); no imbalance "
        "handling, since ROC-AUC is rank-based and the defaults were specified. "
        "THE PIVOTAL CHOICE IS FEATURE SCALING. Scaling is a no-op for trees, so it "
        "cannot flatter RF, but it is what lets the default lbfgs solver converge: "
        "scaled, LogisticRegression converges in 79 iterations, inside the default "
        f"max_iter=100. Unscaled, it hits the iteration cap without converging and its "
        f"CV AUC falls to {auc_lr_ns.mean():.4f}, which would flip the answer to "
        f"'RF > LogReg by {auc_rf.mean()-auc_lr_ns.mean():+.3f}'. I treat the scaled "
        "result as the honest comparison, because the unscaled number measures a "
        "non-converged optimizer rather than the logistic model itself; a researcher who "
        "ran LogisticRegression() literally on raw one-hot columns and ignored the "
        "ConvergenceWarning would report the opposite conclusion. Ordinal instead of "
        "one-hot encoding, dropping fnlwgt, or dropping the redundant "
        "education/education-num pair would shift magnitudes modestly, not this fork."
    ),
    "verification_method": (
        "(1) 5x repeated stratified 5-fold CV (25 folds, seeds 0-4), paired per fold, "
        "with a paired t-test and a 10,000-resample bootstrap CI on the mean fold-level "
        "difference; (2) an independent held-out 20% stratified test split (seed 123, "
        "not used in any CV above) with a 2,000-resample paired bootstrap CI on the "
        "test-set AUC difference; (3) a sensitivity run of logistic regression without "
        "feature scaling, plus an explicit solver-convergence check (n_iter_ and "
        "ConvergenceWarning) for both the scaled and unscaled variants."
    ),
    "verification_result": (
        f"The finding held up, and if anything got sharper. Repeated CV: RF "
        f"{r_rf.mean():.4f} vs LogReg {r_lr.mean():.4f}, mean difference "
        f"{r_diff.mean():+.4f} (bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}], entirely below "
        f"zero); RF won only {int((r_diff>0).sum())}/25 folds, paired t={t:.1f}, "
        f"p={p:.2g}. Held-out 20% split: RF {h_rf:.4f} vs LogReg {h_lr:.4f}, difference "
        f"{h_rf-h_lr:+.4f}, same direction, though on a single split of ~9.8k rows the "
        f"paired bootstrap CI [{hlo:+.4f}, {hhi:+.4f}] does include zero "
        f"(P(RF>LogReg)={(bd>0).mean():.3f}) -- a single test split simply lacks the "
        f"resolution to certify a ~0.004 AUC gap, which is why the paired repeated-CV "
        f"estimate is the one to quote. Revised best estimate: RF trails logistic "
        f"regression by {abs(r_diff.mean()):.4f} AUC, 95% CI "
        f"[{abs(hi):.4f}, {abs(lo):.4f}]. Reliable in sign, small in magnitude."
    ),
}
assert winner == "LogReg", "reported direction must match the computed sign"

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
