"""
H1: Does the choice of model family meaningfully affect predictive performance
on the UCI/OpenML Adult (Census Income) dataset?

Design
------
1. Split 75% "analysis" / 25% "held-out re-test" (stratified, seed 0). The
   re-test split is untouched until the verification stage.
2. Light hyperparameter tuning of each model family on the analysis set
   (3-fold stratified CV, ROC-AUC) so that families are compared at a
   reasonable operating point rather than at arbitrary defaults.
3. Primary analysis: 5-fold stratified CV on the analysis set. Primary metric
   is ROC-AUC (threshold-free, robust to the 24% base rate); accuracy,
   average precision and Brier score reported alongside.
4. Verification: (a) 5x repeated 5-fold CV with seeds 1..5 including paired
   per-fold differences, (b) refit on the whole analysis set and score the
   untouched 25% re-test split, (c) paired bootstrap CI (2000 resamples) of
   the AUC difference on that re-test split.

All models see the identical preprocessing pipeline (median/constant
imputation, standardisation of numerics, one-hot of categoricals) so that
"model family" is the only thing varying. One robustness row uses
HistGradientBoosting with native categorical handling to confirm the encoding
choice is not what drives the result.
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RNG = 0

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")
y = (df["class"] == ">50K").astype(int).values

# fnlwgt is a census sampling weight, not a person-level attribute -> dropped.
# `education` is a redundant string encoding of `education-num` -> dropped.
X = df.drop(columns=["class", "fnlwgt", "education"])

NUM = X.select_dtypes(include=np.number).columns.tolist()
CAT = [c for c in X.columns if c not in NUM]


def make_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                fill_value="Missing")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=10,
                                               sparse_output=False))]), CAT),
    ])


def make_ordinal_pre():
    """Ordinal-coded categoricals for HGB's native categorical support."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                fill_value="Missing")),
                          ("or", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                unknown_value=-1))]), CAT),
    ])


X_an, X_re, y_an, y_re = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RNG)
print(f"analysis n={len(y_an)} (pos {y_an.mean():.3f}) | "
      f"re-test n={len(y_re)} (pos {y_re.mean():.3f})")

# ----------------------------------------------------------------------------
# Model families + small tuning grids
# ----------------------------------------------------------------------------
FAMILIES = {
    "Dummy (prior)": (DummyClassifier(strategy="prior"), {}),
    "GaussianNB": (GaussianNB(), {}),
    "LogisticRegression": (
        LogisticRegression(max_iter=3000),
        {"m__C": [0.03, 0.1, 0.3, 1.0, 3.0]}),
    "kNN": (
        KNeighborsClassifier(n_jobs=-1),
        {"m__n_neighbors": [10, 25, 50, 100], "m__weights": ["uniform", "distance"]}),
    "DecisionTree": (
        DecisionTreeClassifier(random_state=RNG),
        {"m__max_depth": [5, 8, 12, None], "m__min_samples_leaf": [5, 20, 50]}),
    "RandomForest": (
        RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=RNG),
        {"m__min_samples_leaf": [1, 3, 10], "m__max_features": ["sqrt", 0.3]}),
    "HistGradientBoosting": (
        HistGradientBoostingClassifier(random_state=RNG),
        {"m__learning_rate": [0.05, 0.1], "m__max_leaf_nodes": [15, 31, 63],
         "m__l2_regularization": [0.0, 1.0]}),
    "MLP": (
        MLPClassifier(early_stopping=True, max_iter=400, random_state=RNG),
        {"m__hidden_layer_sizes": [(64, 32), (128,)], "m__alpha": [1e-4, 1e-2]}),
}

print("\n=== Stage 2: tuning (3-fold CV on analysis set, ROC-AUC) ===")
tuned = {}
inner = StratifiedKFold(5 if False else 3, shuffle=True, random_state=RNG)
for name, (est, grid) in FAMILIES.items():
    pipe = Pipeline([("p", make_pre()), ("m", est)])
    t0 = time.time()
    if grid:
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1)
        gs.fit(X_an, y_an)
        tuned[name] = gs.best_estimator_
        print(f"{name:22s} cv-auc={gs.best_score_:.4f} {gs.best_params_} "
              f"({time.time()-t0:.0f}s)")
    else:
        tuned[name] = pipe.fit(X_an, y_an)
        print(f"{name:22s} (no tuning) ({time.time()-t0:.0f}s)")

# Robustness row: same family, different encoding.
tuned["HGB (native categorical)"] = Pipeline([
    ("p", make_ordinal_pre()),
    ("m", HistGradientBoostingClassifier(
        random_state=RNG,
        categorical_features=[len(NUM) + i for i in range(len(CAT))]))])


def clone_of(name):
    from sklearn.base import clone
    return clone(tuned[name])


# ----------------------------------------------------------------------------
# Stage 3 + 4: repeated stratified 5-fold CV (seed 0 = primary, 1..5 = check)
# ----------------------------------------------------------------------------
def cv_scores(seeds):
    """Return dict name -> {seed -> [per-fold auc]} plus secondary metrics."""
    auc = {n: {} for n in tuned}
    sec = {n: [] for n in tuned}
    for seed in seeds:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        folds = list(skf.split(X_an, y_an))
        for n in tuned:
            per_fold = []
            for tr, te in folds:
                mdl = clone_of(n).fit(X_an.iloc[tr], y_an[tr])
                p = mdl.predict_proba(X_an.iloc[te])[:, 1]
                yt = y_an[te]
                per_fold.append(roc_auc_score(yt, p))
                if seed == seeds[0]:
                    sec[n].append((accuracy_score(yt, (p >= .5).astype(int)),
                                   average_precision_score(yt, p),
                                   f1_score(yt, (p >= .5).astype(int)),
                                   brier_score_loss(yt, p)))
            auc[n][seed] = per_fold
        print(f"  seed {seed} done")
    return auc, sec


print("\n=== Stage 3/4: 5 x 5-fold CV (seeds 0-5) ===")
SEEDS = [0, 1, 2, 3, 4, 5]
auc, sec = cv_scores(SEEDS)

primary_rows = []
for n in tuned:
    a0 = np.array(auc[n][0])
    allf = np.concatenate([auc[n][s] for s in SEEDS])
    acc, ap, f1, br = np.mean(sec[n], axis=0)
    primary_rows.append(dict(model=n, auc_seed0=a0.mean(), auc_seed0_sd=a0.std(ddof=1),
                             auc_rep=allf.mean(), auc_rep_sd=allf.std(ddof=1),
                             acc=acc, ap=ap, f1=f1, brier=br))
tab = pd.DataFrame(primary_rows).sort_values("auc_rep", ascending=False)
print("\n" + tab.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

REAL = [n for n in tuned if n not in ("Dummy (prior)", "HGB (native categorical)")]
best = max(REAL, key=lambda n: np.mean([np.mean(auc[n][s]) for s in SEEDS]))
worst_real = min(REAL, key=lambda n: np.mean([np.mean(auc[n][s]) for s in SEEDS]))
print(f"\nbest={best}  worst(excl. dummy)={worst_real}")

# Paired per-fold differences across all 30 folds (best vs each competitor).
print("\n=== paired per-fold AUC differences vs best ===")
paired = {}
for n in REAL:
    if n == best:
        continue
    d = np.concatenate([np.array(auc[best][s]) - np.array(auc[n][s]) for s in SEEDS])
    lo, hi = np.percentile(d, [2.5, 97.5])
    paired[n] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)),
                     min=float(d.min()), max=float(d.max()),
                     frac_best_wins=float((d > 0).mean()))
    print(f"{best} - {n:22s}: mean={d.mean():+.4f} sd={d.std(ddof=1):.4f} "
          f"range=[{d.min():+.4f},{d.max():+.4f}] wins={100*(d>0).mean():.0f}%")

HEAD = "LogisticRegression"
d_head = np.concatenate([np.array(auc[best][s]) - np.array(auc[HEAD][s]) for s in SEEDS])
print(f"\nHEADLINE (CV, 30 folds): {best} - {HEAD} = {d_head.mean():+.4f}")

# ----------------------------------------------------------------------------
# Stage 4b/c: untouched held-out re-test split + paired bootstrap CI
# ----------------------------------------------------------------------------
print("\n=== held-out re-test split (never used above) ===")
probs, retest = {}, {}
for n in tuned:
    mdl = clone_of(n).fit(X_an, y_an)
    p = mdl.predict_proba(X_re)[:, 1]
    probs[n] = p
    retest[n] = dict(auc=roc_auc_score(y_re, p),
                     acc=accuracy_score(y_re, (p >= .5).astype(int)),
                     ap=average_precision_score(y_re, p),
                     f1=f1_score(y_re, (p >= .5).astype(int)),
                     brier=brier_score_loss(y_re, p))
rt = pd.DataFrame(retest).T.sort_values("auc", ascending=False)
print(rt.to_string(float_format=lambda v: f"{v:.4f}"))

rng = np.random.default_rng(12345)
idx = np.arange(len(y_re))
boot = {n: [] for n in REAL if n != best}
boot_spread = []
for _ in range(2000):
    b = rng.choice(idx, size=len(idx), replace=True)
    if y_re[b].sum() in (0, len(b)):
        continue
    a_best = roc_auc_score(y_re[b], probs[best][b])
    aucs_b = {n: roc_auc_score(y_re[b], probs[n][b]) for n in REAL}
    for n in boot:
        boot[n].append(a_best - aucs_b[n])
    boot_spread.append(max(aucs_b.values()) - min(aucs_b.values()))

print("\n=== paired bootstrap (2000x) on re-test split ===")
boot_ci = {}
for n, v in boot.items():
    v = np.array(v)
    lo, hi = np.percentile(v, [2.5, 97.5])
    boot_ci[n] = dict(mean=float(v.mean()), lo=float(lo), hi=float(hi),
                      p_best_wins=float((v > 0).mean()))
    print(f"{best} - {n:22s}: {v.mean():+.4f} [{lo:+.4f},{hi:+.4f}] "
          f"P(win)={100*(v>0).mean():.1f}%")
sp = np.array(boot_spread)
print(f"spread(best-worst real family) on re-test: {sp.mean():.4f} "
      f"[{np.percentile(sp,2.5):.4f},{np.percentile(sp,97.5):.4f}]")

primary_value = float(d_head.mean())

out = dict(
    primary_cv_table=tab.to_dict(orient="records"),
    retest_table=rt.reset_index().rename(columns={"index": "model"}).to_dict(orient="records"),
    paired_cv=paired, bootstrap_retest=boot_ci,
    best=best, worst_real=worst_real,
    headline_cv_diff=primary_value,
    headline_retest_diff=float(retest[best]["auc"] - retest[HEAD]["auc"]),
    spread_cv=float(np.mean([np.mean(auc[best][s]) for s in SEEDS])
                    - np.mean([np.mean(auc[worst_real][s]) for s in SEEDS])),
    spread_retest_boot=dict(mean=float(sp.mean()), lo=float(np.percentile(sp, 2.5)),
                            hi=float(np.percentile(sp, 97.5))),
)
with open("detailed_results.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("\nwrote detailed_results.json")
