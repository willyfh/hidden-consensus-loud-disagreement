"""
H4: Does addressing class imbalance improve model quality on the Adult income dataset?

Design
------
Base rate is ~24% positive (>50K) -- mild-to-moderate imbalance.

For two model families (regularized logistic regression, histogram gradient boosting)
we compare five imbalance treatments:
    none          : plain fit on the natural class distribution
    class_weight  : class_weight='balanced'
    undersample   : RandomUnderSampler to 1:1 (train folds only)
    oversample    : RandomOverSampler to 1:1 (train folds only)
    smote         : SMOTE (one-hot space, LR) / SMOTENC (mixed space, HGB)

Each is scored on
    - threshold-free ranking metrics: ROC-AUC, PR-AUC (average precision)
    - fixed-threshold (0.5) metrics: F1, balanced accuracy
    - tuned-threshold metrics: F1 and balanced accuracy at the threshold chosen
      on out-of-fold predictions from the *training* data only.

Stage 1: single stratified 80/20 holdout (seed 0).
Stage 2 (verification): 5 x 5-fold repeated stratified CV over the full data,
         seeds 100..104, with paired bootstrap CIs on the per-fold deltas.
"""

import json
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
RNG = 0

# ---------------------------------------------------------------- data
df = pd.read_csv("adult_income.csv")
df = df.drop_duplicates().reset_index(drop=True)
# fnlwgt is a census sampling weight, not a person-level predictor; 'education'
# is a pure duplicate of the ordinal 'education-num'.
df = df.drop(columns=["fnlwgt", "education"])

y = (df["class"].str.strip() == ">50K").astype(int).to_numpy()
X = df.drop(columns=["class"])
CAT = [c for c in X.columns if X[c].dtype == object]
NUM = [c for c in X.columns if c not in CAT]
X[CAT] = X[CAT].fillna("Missing")  # NaNs are a meaningful "not applicable" level

print(f"n={len(X)}  positives={y.mean():.4f}  cat={len(CAT)} num={len(NUM)}")

CAT_IDX = [X.columns.get_loc(c) for c in CAT]

# ---------------------------------------------------------------- models
def lr_prep():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT),
        ]
    )


def hgb_prep():
    return ColumnTransformer(
        [
            ("num", "passthrough", NUM),
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CAT,
            ),
        ]
    )


HGB_CAT_MASK = [False] * len(NUM) + [True] * len(CAT)


def make_model(family, treatment, seed):
    """Return a pipeline implementing one (model family, imbalance treatment) cell."""
    cw = "balanced" if treatment == "class_weight" else None
    if family == "lr":
        prep = lr_prep()
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight=cw, random_state=seed)
        smote = SMOTE(random_state=seed)  # applied after one-hot encoding
    else:
        prep = hgb_prep()
        clf = HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.1,
            categorical_features=HGB_CAT_MASK,
            class_weight=cw,
            random_state=seed,
        )
        smote = SMOTENC(categorical_features=HGB_CAT_MASK, random_state=seed)

    steps = [("prep", prep)]
    if treatment == "undersample":
        steps.append(("res", RandomUnderSampler(random_state=seed)))
    elif treatment == "oversample":
        steps.append(("res", RandomOverSampler(random_state=seed)))
    elif treatment == "smote":
        steps.append(("res", smote))
    steps.append(("clf", clf))
    return ImbPipeline(steps) if len(steps) == 3 else Pipeline(steps)


FAMILIES = ["lr", "hgb"]
TREATMENTS = ["none", "class_weight", "undersample", "oversample", "smote"]


def best_threshold(y_true, p, metric):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    scorer = f1_score if metric == "f1" else balanced_accuracy_score
    scores = [scorer(y_true, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def score_all(y_true, p, thr_f1, thr_ba):
    return {
        "roc_auc": roc_auc_score(y_true, p),
        "pr_auc": average_precision_score(y_true, p),
        "f1@0.5": f1_score(y_true, (p >= 0.5).astype(int)),
        "bal_acc@0.5": balanced_accuracy_score(y_true, (p >= 0.5).astype(int)),
        "f1@tuned": f1_score(y_true, (p >= thr_f1).astype(int)),
        "bal_acc@tuned": balanced_accuracy_score(y_true, (p >= thr_ba).astype(int)),
    }


def fit_eval(family, treatment, Xtr, ytr, Xte, yte, seed):
    """Fit on train; tune thresholds on inner-CV OOF preds of train; score on test."""
    model = make_model(family, treatment, seed)
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = cross_val_predict(model, Xtr, ytr, cv=inner, method="predict_proba")[:, 1]
    thr_f1 = best_threshold(ytr, oof, "f1")
    thr_ba = best_threshold(ytr, oof, "bal_acc")
    model.fit(Xtr, ytr)
    p = model.predict_proba(Xte)[:, 1]
    out = score_all(yte, p, thr_f1, thr_ba)
    out["thr_f1"] = thr_f1
    out["thr_ba"] = thr_ba
    out["pos_rate@0.5"] = float((p >= 0.5).mean())
    return out


# ================================================================ Stage 1: holdout
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)
rows = []
for fam in FAMILIES:
    for tr in TREATMENTS:
        r = fit_eval(fam, tr, Xtr, ytr, Xte, yte, RNG)
        r.update(family=fam, treatment=tr)
        rows.append(r)
        print(f"[holdout] {fam:4s} {tr:13s} " + " ".join(f"{k}={r[k]:.4f}" for k in
              ["roc_auc", "pr_auc", "f1@0.5", "bal_acc@0.5", "f1@tuned", "bal_acc@tuned"]))

holdout = pd.DataFrame(rows)
holdout.to_csv("holdout_results.csv", index=False)

METRICS = ["roc_auc", "pr_auc", "f1@0.5", "bal_acc@0.5", "f1@tuned", "bal_acc@tuned"]
print("\n[holdout] deltas vs 'none'")
for fam in FAMILIES:
    base = holdout[(holdout.family == fam) & (holdout.treatment == "none")].iloc[0]
    for tr in TREATMENTS[1:]:
        r = holdout[(holdout.family == fam) & (holdout.treatment == tr)].iloc[0]
        print(f"  {fam:4s} {tr:13s} " + " ".join(f"{m}={r[m]-base[m]:+.4f}" for m in METRICS))

# ================================================================ Stage 2: repeated CV
print("\n[verification] 5x5 repeated stratified CV, seeds 100-104")
cv_rows = []
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=100)
for fold_id, (itr, ite) in enumerate(rskf.split(X, y)):
    seed = 100 + fold_id // 5
    Xa, Xb = X.iloc[itr], X.iloc[ite]
    ya, yb = y[itr], y[ite]
    for fam in FAMILIES:
        for tr in TREATMENTS:
            r = fit_eval(fam, tr, Xa, ya, Xb, yb, seed)
            r.update(family=fam, treatment=tr, fold=fold_id, repeat=fold_id // 5)
            cv_rows.append(r)
    print(f"  fold {fold_id + 1}/25 done", flush=True)

cv = pd.DataFrame(cv_rows)
cv.to_csv("cv_results.csv", index=False)


def boot_ci(d, n=10000, seed=7):
    rs = np.random.default_rng(seed)
    d = np.asarray(d, float)
    means = rs.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


print("\n[verification] paired per-fold deltas vs 'none' (mean [95% bootstrap CI])")
summary = {}
for fam in FAMILIES:
    base = cv[(cv.family == fam) & (cv.treatment == "none")].set_index("fold")
    for tr in TREATMENTS[1:]:
        cur = cv[(cv.family == fam) & (cv.treatment == tr)].set_index("fold")
        line = f"  {fam:4s} {tr:13s}"
        for m in METRICS:
            d = (cur[m] - base[m]).to_numpy()
            lo, hi = boot_ci(d)
            summary[f"{fam}|{tr}|{m}"] = {"mean": float(d.mean()), "ci": [lo, hi]}
            line += f" {m}={d.mean():+.4f}[{lo:+.4f},{hi:+.4f}]"
        print(line)

print("\n[verification] absolute means by cell")
print(cv.groupby(["family", "treatment"])[METRICS + ["thr_f1", "pos_rate@0.5"]].mean().round(4))

with open("cv_delta_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# ---------------------------------------------------------------- headline number
hb = summary["hgb|class_weight|pr_auc"]
print(f"\nPRIMARY (HGB, class_weight vs none, PR-AUC delta): "
      f"{hb['mean']:+.4f} CI {hb['ci']}")
