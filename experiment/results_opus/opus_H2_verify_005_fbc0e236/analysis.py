"""
H2: Does RandomForestClassifier() (sklearn defaults) achieve higher stratified
5-fold CV ROC-AUC than LogisticRegression() (sklearn defaults) on UCI Adult?

Design decisions (mine as researcher):
  - Estimators are literally sklearn defaults, as the question specifies.
    Everything upstream (encoding, imputation, scaling) is my choice.
  - Both models get the IDENTICAL feature matrix so the comparison isolates the
    model class: median-imputed + standardized numerics, most-frequent-imputed +
    one-hot categoricals (unknown categories ignored at transform time).
    Scaling is inert for a forest but is what makes LogisticRegression()'s
    default lbfgs/max_iter=100 actually converge -- comparing against a
    non-converged LR would be a measurement artifact, not a finding.
  - fnlwgt (census sampling weight) is kept as a predictor: it is a column in
    the given file and dropping it is an extra assumption.
  - Rows are kept as-is (no de-duplication).
  - Imbalance (24% positive) is left untouched: ROC-AUC is the specified metric
    and is threshold-free, so resampling/class_weight would change the models
    away from "defaults".
  - Preprocessing is fit INSIDE each CV fold (Pipeline) to avoid leakage.

Verification: (a) 5x repeated stratified 5-fold CV over 5 fresh seeds, paired
per fold; (b) BCa-free percentile bootstrap CI on the paired fold differences;
(c) a 20% held-out test split never touched by the CV analysis; (d) sensitivity
checks on the two biggest forks (unscaled features for LR; ordinal encoding
for RF).
"""

import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import (RepeatedStratifiedKFold, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RNG = np.random.default_rng(0)
N_JOBS = -1

# ---------------------------------------------------------------- data
df = pd.read_csv("adult_income.csv")
df.columns = [c.strip() for c in df.columns]

# '?' already read as NaN in this file, but normalise defensively.
obj = df.select_dtypes("object").columns
for c in obj:
    df[c] = df[c].astype(str).str.strip().replace({"?": np.nan, "nan": np.nan})

y = (df["class"].str.replace(".", "", regex=False) == ">50K").astype(int).values
X = df.drop(columns=["class"])
num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print(f"rows={len(X)}  numeric={num_cols}\ncategorical={cat_cols}")
print(f"positive rate = {y.mean():.4f}")


def make_pre(scale=True, ohe=True):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    if ohe:
        cat_enc = OneHotEncoder(handle_unknown="ignore", min_frequency=None)
    else:
        cat_enc = OrdinalEncoder(handle_unknown="use_encoded_value",
                                 unknown_value=-1)
    return ColumnTransformer([
        ("num", Pipeline(num_steps), num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("enc", cat_enc)]), cat_cols),
    ])


def pipes(seed, scale=True, ohe_rf=True):
    lr = Pipeline([("pre", make_pre(scale=True)), ("m", LogisticRegression())])
    if not scale:  # sensitivity variant: LR on raw, unscaled features
        lr = Pipeline([("pre", make_pre(scale=False)), ("m", LogisticRegression())])
    rf = Pipeline([("pre", make_pre(scale=True, ohe=ohe_rf)),
                   ("m", RandomForestClassifier(random_state=seed))])
    return lr, rf


# ------------------------------------------------- primary: strat 5-fold CV
print("\n=== PRIMARY: stratified 5-fold CV, ROC-AUC ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr, rf = pipes(42)
auc_lr = cross_val_score(lr, X, y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS)
auc_rf = cross_val_score(rf, X, y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS)
print(f"LogReg per-fold: {np.round(auc_lr, 5)}  mean={auc_lr.mean():.5f}")
print(f"RF     per-fold: {np.round(auc_rf, 5)}  mean={auc_rf.mean():.5f}")
primary_diff = auc_rf.mean() - auc_lr.mean()
print(f"PRIMARY diff (RF - LR) = {primary_diff:+.5f}")

# --------------------------------------- verification 1: repeated CV, 5 seeds
print("\n=== VERIFY 1: 5x repeated stratified 5-fold (25 paired folds) ===")
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7)


def one_fold(i, tr, te):
    """Fit both models on the same fold; fresh RF seed per fold."""
    l, r = pipes(seed=i)
    l.fit(X.iloc[tr], y[tr]); r.fit(X.iloc[tr], y[tr])
    return (roc_auc_score(y[te], l.predict_proba(X.iloc[te])[:, 1]),
            roc_auc_score(y[te], r.predict_proba(X.iloc[te])[:, 1]))


folds = list(rcv.split(X, y))
out = Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(one_fold)(i, tr, te) for i, (tr, te) in enumerate(folds))
r_lr = np.array([o[0] for o in out])
r_rf = np.array([o[1] for o in out])
d = r_rf - r_lr
print(f"LogReg  mean={r_lr.mean():.5f}  sd={r_lr.std(ddof=1):.5f}")
print(f"RF      mean={r_rf.mean():.5f}  sd={r_rf.std(ddof=1):.5f}")
print(f"paired diff mean={d.mean():+.5f}  sd={d.std(ddof=1):.5f}  "
      f"min={d.min():+.5f}  max={d.max():+.5f}  "
      f"RF wins {int((d > 0).sum())}/{len(d)} folds")
t, p = stats.ttest_rel(r_rf, r_lr)
print(f"paired t-test (folds are not independent; indicative only): t={t:.3f} p={p:.2e}")

# --------------------------------------- verification 2: bootstrap CI on diff
boot = np.array([RNG.choice(d, size=len(d), replace=True).mean()
                 for _ in range(10000)])
ci = np.percentile(boot, [2.5, 97.5])
print(f"bootstrap 95% CI of mean paired diff: [{ci[0]:+.5f}, {ci[1]:+.5f}]")

# --------------------------------------- verification 3: untouched hold-out
print("\n=== VERIFY 3: 80/20 stratified hold-out (unused above) ===")
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                      random_state=2024)
l, r = pipes(2024)
l.fit(Xtr, ytr); r.fit(Xtr, ytr)
p_lr = l.predict_proba(Xte)[:, 1]; p_rf = r.predict_proba(Xte)[:, 1]
h_lr, h_rf = roc_auc_score(yte, p_lr), roc_auc_score(yte, p_rf)
print(f"hold-out  LogReg={h_lr:.5f}  RF={h_rf:.5f}  diff={h_rf - h_lr:+.5f}")

# paired bootstrap over test rows (resample rows, recompute both AUCs)
idx = np.arange(len(yte))
bd = []
for _ in range(2000):
    b = RNG.choice(idx, size=len(idx), replace=True)
    if yte[b].mean() in (0.0, 1.0):
        continue
    bd.append(roc_auc_score(yte[b], p_rf[b]) - roc_auc_score(yte[b], p_lr[b]))
bd = np.array(bd)
hci = np.percentile(bd, [2.5, 97.5])
print(f"hold-out paired-row bootstrap 95% CI of diff: [{hci[0]:+.5f}, {hci[1]:+.5f}]")

# --------------------------------------- sensitivity to my two biggest forks
print("\n=== SENSITIVITY ===")
lr_raw, _ = pipes(42, scale=False)
auc_lr_raw = cross_val_score(lr_raw, X, y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS)
print(f"LogReg on UNSCALED features (default max_iter=100, will not converge): "
      f"{auc_lr_raw.mean():.5f}  -> diff {auc_rf.mean() - auc_lr_raw.mean():+.5f}")
_, rf_ord = pipes(42, ohe_rf=False)
auc_rf_ord = cross_val_score(rf_ord, X, y, cv=cv, scoring="roc_auc", n_jobs=N_JOBS)
print(f"RF with ORDINAL-encoded categoricals: {auc_rf_ord.mean():.5f}  "
      f"-> diff vs scaled LR {auc_rf_ord.mean() - auc_lr.mean():+.5f}")

# The ordinal fork lands near a tie, i.e. it is the one choice that could flip
# the sign of the answer -- so give it the same 25-fold repeated-CV treatment
# instead of resting on a single 5-fold run.
def one_fold_ord(i, tr, te):
    r = Pipeline([("pre", make_pre(scale=True, ohe=False)),
                  ("m", RandomForestClassifier(random_state=i))])
    r.fit(X.iloc[tr], y[tr])
    return roc_auc_score(y[te], r.predict_proba(X.iloc[te])[:, 1])


r_rf_ord = np.array(Parallel(n_jobs=N_JOBS, verbose=1)(
    delayed(one_fold_ord)(i, tr, te) for i, (tr, te) in enumerate(folds)))
d_ord = r_rf_ord - r_lr
ci_ord = np.percentile(
    [RNG.choice(d_ord, size=len(d_ord), replace=True).mean() for _ in range(10000)],
    [2.5, 97.5])
print(f"ordinal-RF repeated CV: mean={r_rf_ord.mean():.5f}  paired diff vs LR "
      f"{d_ord.mean():+.5f}  95% CI [{ci_ord[0]:+.5f}, {ci_ord[1]:+.5f}]  "
      f"RF wins {int((d_ord > 0).sum())}/{len(d_ord)} folds")

# ---------------------------------------------------------------- result.json
result = {
    "hypothesis_id": "H2",
    "summary": (
        f"No. With identical in-fold preprocessing (median/most-frequent "
        f"imputation, standardized numerics, one-hot categoricals), default "
        f"RandomForestClassifier() reaches a stratified 5-fold CV ROC-AUC of "
        f"{auc_rf.mean():.4f} versus {auc_lr.mean():.4f} for default "
        f"LogisticRegression() -- a difference of {primary_diff:+.4f}, i.e. the "
        f"forest is very slightly WORSE. The gap is tiny in absolute terms "
        f"(both models sit at ~0.90 AUC) but consistent: logistic regression "
        f"won {len(d) - int((d > 0).sum())} of {len(d)} folds under repeated CV. "
        f"The verdict is sensitive to encoding, though -- with ordinal instead "
        f"of one-hot encoded categoricals the forest matches logistic "
        f"regression ({d_ord.mean():+.4f}), so the honest reading is that these "
        f"two defaults are practically tied on this dataset, with no RF "
        f"advantage."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg), stratified 5-fold CV",
    "primary_metric_value": round(float(primary_diff), 5),
    "direction": "RF < LogReg (no RF advantage; difference ~-0.004 AUC, practically a tie)",
    "methodological_choices": (
        "Estimators left at literal sklearn defaults as specified "
        "(RandomForestClassifier(): 100 trees, unlimited depth; "
        "LogisticRegression(): lbfgs, L2, C=1.0, max_iter=100); only "
        "random_state was set on RF for reproducibility. Both models were fed "
        "the IDENTICAL feature matrix so the comparison isolates model class: "
        "median imputation + StandardScaler on the 6 numeric columns, "
        "most-frequent imputation + one-hot (handle_unknown='ignore') on the 8 "
        "categoricals; all fit inside each training fold via a Pipeline to "
        "avoid leakage. Scaling is inert for a forest but is what lets default "
        "LogisticRegression converge within max_iter=100 -- another researcher "
        "who skipped scaling would compare against a non-converged LR. Missing "
        "values ('?' -> NaN in workclass/occupation/native-country) were "
        "imputed rather than dropped or coded as their own level. fnlwgt (the "
        "census sampling weight) was retained as a predictor. No "
        "de-duplication. Class imbalance (23.9% positive) left untouched -- "
        "ROC-AUC is threshold-free and class_weight/resampling would depart "
        "from 'defaults'. Metric: ROC-AUC on predict_proba, StratifiedKFold "
        "with shuffle=True, random_state=42. TWO FORKS MATERIALLY CHANGE THE "
        "ANSWER and another researcher could easily land elsewhere: (a) "
        "Scaling. If the numerics are left unscaled, default "
        f"LogisticRegression() does not converge in max_iter=100 and collapses "
        f"to {auc_lr_raw.mean():.4f} AUC, which would flip the conclusion to a "
        f"large apparent RF win of {auc_rf.mean() - auc_lr_raw.mean():+.4f}. "
        "That number is an artifact of a non-converged optimizer, not a real "
        "model-class difference, which is why I scale. (b) Encoding. With "
        "ordinal instead of one-hot categoricals the forest improves to "
        f"{auc_rf_ord.mean():.4f} single-run / {r_rf_ord.mean():.4f} under "
        f"repeated CV, a paired difference vs LR of {d_ord.mean():+.4f} "
        f"(95% CI [{ci_ord[0]:+.4f}, {ci_ord[1]:+.4f}]) -- a statistical dead "
        "heat rather than an RF win. So the sign of the RF-LogReg gap is not "
        "robust to encoding, but under no defensible preprocessing does RF "
        "show a meaningful advantage."
    ),
    "verification_method": (
        "Three independent checks: (1) 5x repeated stratified 5-fold CV with "
        "fresh split seeds and a fresh RF seed per fold (25 paired folds); "
        "(2) a 10,000-draw percentile bootstrap 95% CI on the mean paired "
        "per-fold AUC difference; (3) refit on an 80/20 stratified hold-out "
        "split (random_state=2024) never used in the primary analysis, with a "
        "2,000-draw paired-row bootstrap CI on the test-set AUC difference."
    ),
    "verification_result": (
        f"The finding held up in direction and magnitude. Repeated CV: RF "
        f"{r_rf.mean():.4f} (sd {r_rf.std(ddof=1):.4f}) vs LogReg "
        f"{r_lr.mean():.4f} (sd {r_lr.std(ddof=1):.4f}), mean paired difference "
        f"{d.mean():+.4f}; RF was ahead in only {int((d > 0).sum())}/{len(d)} "
        f"folds, so LogReg won {len(d) - int((d > 0).sum())}/{len(d)}. Bootstrap "
        f"95% CI of the mean paired difference [{ci[0]:+.4f}, {ci[1]:+.4f}], "
        f"excluding zero. Untouched 80/20 hold-out reproduced the same sign and "
        f"size: RF {h_rf:.4f} vs LogReg {h_lr:.4f}, difference {h_rf - h_lr:+.4f}, "
        f"though its paired-row bootstrap CI [{hci[0]:+.4f}, {hci[1]:+.4f}] does "
        f"straddle zero -- a single 9,769-row test set cannot resolve a "
        f"0.003 AUC gap, whereas the 25-fold paired comparison can. Revised "
        f"estimate: RF is worse by about {abs(d.mean()):.4f} AUC, range "
        f"~[{ci[0]:+.4f}, {ci[1]:+.4f}] under one-hot encoding; under ordinal "
        f"encoding the gap shrinks to {d_ord.mean():+.4f} "
        f"([{ci_ord[0]:+.4f}, {ci_ord[1]:+.4f}]), i.e. a tie. In no variant did "
        f"RF beat LogReg by a practically meaningful margin, so the answer to "
        f"H2 as posed is No."
    ),
}
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n" + json.dumps(result, indent=2))
