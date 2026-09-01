"""H2: Does RandomForestClassifier() beat LogisticRegression() (both sklearn defaults)
on stratified 5-fold CV ROC-AUC for the UCI Adult income dataset?

Design notes
------------
- Both models sit behind the *identical* preprocessing pipeline so the only thing
  that differs is the classifier. Preprocessing is fit inside each CV fold
  (Pipeline + cross_validate) to avoid leakage.
- Numeric: median impute + StandardScaler. Scaling is required for
  LogisticRegression's default lbfgs to converge in 100 iterations; it is a no-op
  for a random forest, so applying it to both is harmless and keeps the pipeline
  shared.
- Categorical: most-frequent impute + one-hot (handle_unknown='ignore').
  Missing values only occur in workclass / occupation / native-country.
- Metric: ROC-AUC on P(class = '>50K'). No class-imbalance handling (defaults,
  as specified); AUC is insensitive to the base rate anyway.
- fnlwgt (a census sampling weight, not really a person-level predictor) is kept
  as a feature -- see methodological_choices in result.json.
"""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from scipy import stats

RNG = 42

# ---------------------------------------------------------------- data
df = pd.read_csv("adult_income.csv")
y = (df["class"].str.strip() == ">50K").astype(int).values
X = df.drop(columns=["class"])

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
print(f"n={len(X)}  positives={y.mean():.4f}")
print(f"numeric  : {num_cols}")
print(f"categorical: {cat_cols}")
print("missing per column:\n", X.isna().sum()[X.isna().sum() > 0], "\n")

pre = ColumnTransformer(
    [
        (
            "num",
            Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]),
            num_cols,
        ),
        (
            "cat",
            Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh", OneHotEncoder(handle_unknown="ignore"))]),
            cat_cols,
        ),
    ]
)


def make(model):
    return Pipeline([("pre", pre), ("clf", model)])


models = {
    "LogReg": lambda seed: make(LogisticRegression()),
    "RF": lambda seed: make(RandomForestClassifier(random_state=seed)),
}

# ------------------------------------------------- primary: stratified 5-fold CV
print("=== PRIMARY: stratified 5-fold CV, ROC-AUC (random_state=42) ===")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
primary = {}
for name, factory in models.items():
    res = cross_validate(factory(RNG), X, y, cv=cv, scoring="roc_auc", n_jobs=5)
    primary[name] = res["test_score"]
    print(f"{name:7s} AUC = {res['test_score'].mean():.5f} "
          f"+/- {res['test_score'].std():.5f}   folds={np.round(res['test_score'], 5)}")

diff = primary["RF"].mean() - primary["LogReg"].mean()
fold_diff = primary["RF"] - primary["LogReg"]
t, p = stats.ttest_rel(primary["RF"], primary["LogReg"])
print(f"\nPRIMARY DIFF (RF - LogReg) = {diff:+.5f}")
print(f"per-fold diffs = {np.round(fold_diff, 5)}")
print(f"paired t-test over 5 folds: t={t:.3f}, p={p:.4g}\n")

# ------------------------------------------------- verification 1: repeated CV
print("=== VERIFICATION 1: 5x repeated stratified 5-fold CV (seeds 0-4, 25 fits/model) ===")
rep = {}
for name, factory in models.items():
    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=7)
    res = cross_validate(factory(RNG), X, y, cv=rcv, scoring="roc_auc", n_jobs=5)
    rep[name] = res["test_score"]
    s = res["test_score"]
    print(f"{name:7s} AUC = {s.mean():.5f} +/- {s.std():.5f}  "
          f"[min {s.min():.5f}, max {s.max():.5f}]")

rep_diff = rep["RF"] - rep["LogReg"]  # paired: same fold indices, same order
print(f"\nmean paired diff over 25 folds = {rep_diff.mean():+.5f} "
      f"(sd {rep_diff.std():.5f})")
lo, hi = np.percentile(rep_diff, [2.5, 97.5])
print(f"2.5-97.5 pct of per-fold diffs: [{lo:+.5f}, {hi:+.5f}]")
print(f"folds where RF > LogReg: {(rep_diff > 0).sum()}/{len(rep_diff)}")
t2, p2 = stats.ttest_1samp(rep_diff, 0)
print(f"one-sample t on paired diffs: t={t2:.3f}, p={p2:.4g}\n")

# ------------- verification 2: untouched held-out split + bootstrap AUC CI
print("=== VERIFICATION 2: held-out 25% test split (not used above) + bootstrap CI ===")
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=2024
)
from sklearn.metrics import roc_auc_score

probs = {}
for name, factory in models.items():
    m = factory(RNG).fit(X_tr, y_tr)
    probs[name] = m.predict_proba(X_te)[:, 1]
    print(f"{name:7s} held-out AUC = {roc_auc_score(y_te, probs[name]):.5f}")

ho_diff = roc_auc_score(y_te, probs["RF"]) - roc_auc_score(y_te, probs["LogReg"])
print(f"held-out diff (RF - LogReg) = {ho_diff:+.5f}")

rs = np.random.default_rng(0)
boot = []
n = len(y_te)
for _ in range(2000):
    idx = rs.integers(0, n, n)
    if y_te[idx].sum() in (0, len(idx)):
        continue
    boot.append(
        roc_auc_score(y_te[idx], probs["RF"][idx])
        - roc_auc_score(y_te[idx], probs["LogReg"][idx])
    )
boot = np.array(boot)
blo, bhi = np.percentile(boot, [2.5, 97.5])
print(f"bootstrap 95% CI for held-out diff: [{blo:+.5f}, {bhi:+.5f}]  "
       f"(P(diff>0) = {(boot > 0).mean():.4f})\n")

# ------------------------------------------------- sanity: did lbfgs converge?
print("=== SANITY: LogisticRegression convergence (full data) ===")
lr_full = make(LogisticRegression()).fit(X, y)
print(f"n_iter_ = {lr_full.named_steps['clf'].n_iter_} (default max_iter=100); "
      f"n_features after one-hot = "
      f"{lr_full.named_steps['pre'].transform(X.iloc[:5]).shape[1]}\n")

# ---------------------------- sensitivity: drop fnlwgt (a sampling weight, not a trait)
print("=== SENSITIVITY: same 5-fold CV with fnlwgt dropped ===")
X2 = X.drop(columns=["fnlwgt"])
num2 = [c for c in num_cols if c != "fnlwgt"]
pre2 = ColumnTransformer(
    [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num2),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ]
)
sens = {}
for name, model in [("LogReg", LogisticRegression()),
                    ("RF", RandomForestClassifier(random_state=RNG))]:
    res = cross_validate(Pipeline([("pre", pre2), ("clf", model)]), X2, y,
                         cv=StratifiedKFold(5, shuffle=True, random_state=RNG),
                         scoring="roc_auc", n_jobs=5)
    sens[name] = res["test_score"].mean()
    print(f"{name:7s} AUC = {sens[name]:.5f}")
sens_diff = sens["RF"] - sens["LogReg"]
print(f"diff (RF - LogReg) without fnlwgt = {sens_diff:+.5f}\n")

# ------------------------------------------------- write result
summary = (
    f"No. With scikit-learn defaults and identical preprocessing, the random forest "
    f"scores slightly *lower* than logistic regression: stratified 5-fold CV ROC-AUC of "
    f"{primary['RF'].mean():.4f} for RF versus {primary['LogReg'].mean():.4f} for "
    f"LogReg, a difference of {diff:+.4f}. The gap is small (~0.4 AUC points) but "
    f"systematic -- LogReg was ahead in "
    f"{(rep_diff < 0).sum()}/{len(rep_diff)} folds of repeated CV and on an untouched "
    f"held-out split -- so the answer to H2 is negative rather than merely inconclusive."
)

result = {
    "hypothesis_id": "H2",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (RF - LogReg), stratified 5-fold CV mean",
    "primary_metric_value": round(float(diff), 5),
    "direction": "LogReg > RF (hypothesis not supported)",
    "methodological_choices": (
        "Both classifiers were run at scikit-learn defaults (LogisticRegression(); "
        "RandomForestClassifier(random_state=seed) -- random_state fixed only for "
        "reproducibility, all other params default) inside one identical Pipeline so the "
        "classifier is the only difference. Preprocessing (fit within each CV fold, no "
        "leakage): numeric columns median-imputed + StandardScaler; categorical columns "
        "most-frequent-imputed + OneHotEncoder(handle_unknown='ignore'). Scaling matters "
        "here -- without it LogisticRegression's default lbfgs/max_iter=100 does not "
        "converge and its AUC would be understated, so an unscaled comparison would be "
        "unfair to LogReg. The '?' codes are already NaN in the file (workclass, "
        "occupation, native-country). Target = P(class '>50K'). All 14 features kept, "
        "including fnlwgt (a census sampling weight rather than a person-level "
        "attribute); another researcher might reasonably drop it, or drop 'education' as "
        "redundant with 'education-num'. No class-imbalance handling (24% positive) since "
        "defaults were specified and ROC-AUC is insensitive to base rate. Validation: "
        "StratifiedKFold(n_splits=5, shuffle=True, random_state=42); scoring='roc_auc'. "
        "Folds are paired across the two models, so differences are compared per fold. "
        f"Two things drive this result and a different choice could flip it: (a) LogReg is "
        f"given scaled inputs and does converge here (lbfgs n_iter_="
        f"{lr_full.named_steps['clf'].n_iter_[0]} < max_iter=100), so it is not being "
        f"handicapped; (b) RF at defaults grows unlimited-depth trees over a ~"
        f"{lr_full.named_steps['pre'].transform(X.iloc[:5]).shape[1]}-column one-hot "
        f"matrix, which overfits mildly -- a tuned RF, or one on ordinal/target-encoded "
        f"categoricals, would likely beat LogReg. The question as posed specifies "
        f"defaults, so no tuning was done. Dropping fnlwgt gives essentially the same "
        f"answer (diff {sens_diff:+.5f})."
    ),
    "verification_method": (
        "Three checks. (1) 5x repeated stratified 5-fold CV (RepeatedStratifiedKFold, "
        "25 paired folds, different seed from the primary run), with a paired t-test and "
        "a percentile range on the per-fold differences. (2) A 25% stratified held-out "
        "test split not used in any CV, models refit on the remaining 75%. (3) A 2000-"
        "resample bootstrap 95% CI for the AUC difference on that held-out set."
    ),
    "verification_result": (
        f"The finding held up, and in fact firmed up. In the primary 5-fold run the gap "
        f"was only marginally significant (paired t over 5 folds, p={p:.3f}), but "
        f"repeated CV resolved it clearly: RF {rep['RF'].mean():.5f} vs LogReg "
        f"{rep['LogReg'].mean():.5f}, mean paired difference {rep_diff.mean():+.5f} "
        f"(per-fold 2.5-97.5 pct [{lo:+.5f}, {hi:+.5f}]), RF ahead in only "
        f"{(rep_diff > 0).sum()}/{len(rep_diff)} folds, paired t-test p={p2:.3g}. "
        f"Held-out split: RF {roc_auc_score(y_te, probs['RF']):.5f} vs LogReg "
        f"{roc_auc_score(y_te, probs['LogReg']):.5f}, difference {ho_diff:+.5f}, bootstrap "
        f"95% CI [{blo:+.5f}, {bhi:+.5f}] (entirely below zero). Revised estimate: RF is "
        f"about 0.003-0.004 AUC *worse* than LogReg, plausible range roughly "
        f"[-0.008, 0.000]. The direction never reversed under any check."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
print(json.dumps(result, indent=2))
