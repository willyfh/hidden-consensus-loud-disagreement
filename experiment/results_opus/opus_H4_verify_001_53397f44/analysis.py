"""
H4: Does addressing class imbalance improve model quality on the Adult income dataset?

Design
------
Class balance: 23.9% >50K vs 76.1% <=50K (3.18:1) -- moderate imbalance.

We compare five "imbalance handling" conditions for each of two model families:
    none            : plain fit on the natural class distribution
    class_weight    : class_weight='balanced'
    undersample     : RandomUnderSampler to 1:1 (train folds only)
    oversample      : RandomOverSampler to 1:1 (train folds only)
    smotenc         : SMOTE-NC synthetic minority oversampling to 1:1 (train folds only)

Model families:
    LogReg : one-hot + standardized numerics -> LogisticRegression(max_iter=2000)
    HGB    : ordinal-encoded categoricals -> HistGradientBoostingClassifier (native cat support)

Metrics (a deliberate mix of threshold-free and threshold-dependent):
    ROC-AUC, PR-AUC(average precision), Brier score  -- ranking / calibration, threshold free
    accuracy, F1, balanced accuracy at the default 0.5 threshold
    F1 at a threshold tuned on TRAINING data only (out-of-fold), applied to the test set

The last one is the crux: resampling/reweighting mostly moves the operating point.
Tuning the decision threshold on an unmodified model is the control condition that
tells us whether imbalance handling adds anything beyond threshold choice.

Verification: 5x5 repeated stratified CV (seeds 0..4) on the full dataset,
paired per-fold differences vs. the 'none' baseline + a held-out re-test split
(seed 2024) that was not used for the initial analysis.
"""

import json
import warnings

import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler, SMOTENC
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import (StratifiedKFold, cross_val_predict,
                                     train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
RNG = 42

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv", skipinitialspace=True, na_values=["?"])
df = df.drop_duplicates().reset_index(drop=True)
# fnlwgt is a census sampling weight, not a property of the person -> drop
df = df.drop(columns=["fnlwgt"])

y = (df["class"] == ">50K").astype(int).to_numpy()
X = df.drop(columns=["class"]).copy()

CAT = [c for c in X.columns if X[c].dtype == object]
NUM = [c for c in X.columns if c not in CAT]
# constant-fill missing categoricals up front: '?' is informative non-response,
# and SMOTE-NC cannot handle NaN categories.
X[CAT] = X[CAT].fillna("Missing")
X = X[NUM + CAT]  # fixed column order so integer indices are stable
NUM_IDX = list(range(len(NUM)))
CAT_IDX = list(range(len(NUM), len(NUM) + len(CAT)))

print(f"n={len(X)}  positives={y.mean():.4f}  ratio=1:{(1-y.mean())/y.mean():.2f}")
print(f"numeric={NUM}\ncategorical={CAT}\n")


# ----------------------------------------------------------------------------
# Model / condition factory
# ----------------------------------------------------------------------------
def make_pipeline(model_name, condition, seed=RNG):
    """Build an (optionally resampling) pipeline. Samplers live INSIDE the
    pipeline so they only ever touch training folds."""
    balanced = condition == "class_weight"

    if model_name == "LogReg":
        prep = ColumnTransformer(
            [("num", StandardScaler(), NUM_IDX),
             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT_IDX)]
        )
        clf = LogisticRegression(
            max_iter=2000, C=1.0, solver="lbfgs",
            class_weight="balanced" if balanced else None, random_state=seed,
        )
    else:  # HGB
        prep = ColumnTransformer(
            [("num", "passthrough", NUM_IDX),
             ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                    unknown_value=-1), CAT_IDX)]
        )
        clf = HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.1, max_leaf_nodes=31,
            early_stopping=False,
            categorical_features=np.array(
                [False] * len(NUM_IDX) + [True] * len(CAT_IDX)),
            class_weight="balanced" if balanced else None, random_state=seed,
        )

    steps = []
    if condition == "undersample":
        steps.append(("samp", RandomUnderSampler(random_state=seed)))
    elif condition == "oversample":
        steps.append(("samp", RandomOverSampler(random_state=seed)))
    elif condition == "smotenc":
        steps.append(("samp", SMOTENC(categorical_features=CAT_IDX,
                                      k_neighbors=5, random_state=seed)))
    steps += [("prep", prep), ("clf", clf)]
    return ImbPipeline(steps) if len(steps) == 3 else Pipeline(steps)


CONDITIONS = ["none", "class_weight", "undersample", "oversample", "smotenc"]
MODELS = ["LogReg", "HGB"]


def best_f1_threshold(y_true, p):
    """Threshold maximising F1, searched on a fine grid."""
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 197)))
    scores = [f1_score(y_true, (p >= t).astype(int)) for t in grid]
    return float(grid[int(np.argmax(scores))])


def evaluate(y_true, p, thr_tuned):
    yh = (p >= 0.5).astype(int)
    yt = (p >= thr_tuned).astype(int)
    return dict(
        roc_auc=roc_auc_score(y_true, p),
        pr_auc=average_precision_score(y_true, p),
        brier=brier_score_loss(y_true, p),
        accuracy=accuracy_score(y_true, yh),
        f1=f1_score(y_true, yh),
        precision=precision_score(y_true, yh, zero_division=0),
        recall=recall_score(y_true, yh),
        bal_acc=balanced_accuracy_score(y_true, yh),
        f1_tuned=f1_score(y_true, yt),
        acc_tuned=accuracy_score(y_true, yt),
        bal_acc_tuned=balanced_accuracy_score(y_true, yt),
        thr=thr_tuned,
    )


# ----------------------------------------------------------------------------
# Stage 1: primary analysis on a single stratified 70/30 split
# ----------------------------------------------------------------------------
def run_split(seed, label):
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed)
    rows = []
    for m in MODELS:
        for c in CONDITIONS:
            pipe = make_pipeline(m, c, seed=seed)
            # threshold tuned on out-of-fold TRAIN predictions only (no test leakage)
            oof = cross_val_predict(
                pipe, Xtr, ytr, cv=StratifiedKFold(5, shuffle=True, random_state=seed),
                method="predict_proba", n_jobs=-1)[:, 1]
            thr = best_f1_threshold(ytr, oof)
            pipe.fit(Xtr, ytr)
            p = pipe.predict_proba(Xte)[:, 1]
            r = evaluate(yte, p, thr)
            r.update(model=m, condition=c, split=label)
            rows.append(r)
            print(f"[{label}] {m:6s} {c:13s} AUC={r['roc_auc']:.4f} "
                  f"PR={r['pr_auc']:.4f} acc={r['accuracy']:.4f} "
                  f"F1={r['f1']:.4f} F1*={r['f1_tuned']:.4f} thr*={thr:.3f} "
                  f"brier={r['brier']:.4f}")
    return pd.DataFrame(rows)


print("=" * 88)
print("STAGE 1: primary holdout analysis (stratified 70/30, seed 42)")
print("=" * 88)
main = run_split(RNG, "primary")

# ----------------------------------------------------------------------------
# Stage 2: verification A -- 5x5 repeated stratified CV, paired per-fold deltas
# ----------------------------------------------------------------------------
print("\n" + "=" * 88)
print("STAGE 2: verification -- 5x5 repeated stratified CV (seeds 0-4), paired deltas")
print("=" * 88)

cv_rows = []
for seed in range(5):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(skf.split(X, y)):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        # one threshold per (model, condition), tuned inside the training fold
        for m in MODELS:
            for c in CONDITIONS:
                pipe = make_pipeline(m, c, seed=seed)
                inner = cross_val_predict(
                    pipe, Xtr, ytr,
                    cv=StratifiedKFold(3, shuffle=True, random_state=seed),
                    method="predict_proba", n_jobs=-1)[:, 1]
                thr = best_f1_threshold(ytr, inner)
                pipe.fit(Xtr, ytr)
                p = pipe.predict_proba(Xte)[:, 1]
                r = evaluate(yte, p, thr)
                r.update(model=m, condition=c, seed=seed, fold=fold)
                cv_rows.append(r)
    print(f"  ... seed {seed} done")

cv = pd.DataFrame(cv_rows)
cv.to_csv("cv_results.csv", index=False)

METRICS = ["roc_auc", "pr_auc", "accuracy", "f1", "bal_acc", "f1_tuned",
           "acc_tuned", "bal_acc_tuned", "brier"]

print("\nMean over 25 folds:")
print(cv.groupby(["model", "condition"])[METRICS].mean().round(4).to_string())


def paired_delta(model, cond, metric):
    """Paired (cond - none) difference across the 25 identical folds."""
    key = ["seed", "fold"]
    a = cv[(cv.model == model) & (cv.condition == cond)].set_index(key)[metric]
    b = cv[(cv.model == model) & (cv.condition == "none")].set_index(key)[metric]
    d = (a - b).dropna().to_numpy()
    boot = np.array([np.mean(np.random.default_rng(i).choice(d, len(d), replace=True))
                     for i in range(5000)])
    return d.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5), \
        float((d > 0).mean())


print("\nPaired deltas vs. 'none' (mean [95% bootstrap CI], win-rate over 25 folds):")
delta_rows = []
for m in MODELS:
    for c in CONDITIONS[1:]:
        for metric in ["roc_auc", "pr_auc", "accuracy", "f1", "bal_acc",
                       "f1_tuned", "brier"]:
            mu, lo, hi, wr = paired_delta(m, c, metric)
            delta_rows.append(dict(model=m, condition=c, metric=metric,
                                   delta=mu, lo=lo, hi=hi, win_rate=wr))
            sig = "*" if (lo > 0) or (hi < 0) else " "
            print(f"  {m:6s} {c:13s} {metric:9s} {mu:+.5f} "
                  f"[{lo:+.5f},{hi:+.5f}]{sig} win={wr:.2f}")
deltas = pd.DataFrame(delta_rows)
deltas.to_csv("paired_deltas.csv", index=False)

# ----------------------------------------------------------------------------
# Stage 3: verification B -- fresh held-out re-test split, unused until now
# ----------------------------------------------------------------------------
print("\n" + "=" * 88)
print("STAGE 3: verification -- independent re-test split (seed 2024)")
print("=" * 88)
retest = run_split(2024, "retest")

all_holdout = pd.concat([main, retest], ignore_index=True)
all_holdout.to_csv("holdout_results.csv", index=False)

# ----------------------------------------------------------------------------
# Headline numbers
# ----------------------------------------------------------------------------
print("\n" + "=" * 88)
print("HEADLINE")
print("=" * 88)

# Primary metric: mean paired ΔROC-AUC of the best imbalance treatment vs none,
# averaged over both model families, from the 25-fold repeated CV.
prim = deltas[(deltas.metric == "roc_auc")]
print("\nΔROC-AUC by condition (avg over both models):")
print(prim.groupby("condition")[["delta", "lo", "hi"]].mean().round(5).to_string())

best_auc_cond = prim.groupby("condition")["delta"].mean().idxmax()
best_auc_val = prim.groupby("condition")["delta"].mean().max()

cw = deltas[(deltas.condition == "class_weight") & (deltas.metric == "roc_auc")]
cw_mean = cw.delta.mean()
print(f"\nclass_weight ΔROC-AUC: {cw_mean:+.5f} "
      f"(LogReg {cw[cw.model=='LogReg'].delta.iloc[0]:+.5f}, "
      f"HGB {cw[cw.model=='HGB'].delta.iloc[0]:+.5f})")

# Best single imbalance treatment on ROC-AUC across all model x condition cells
best_cell = prim.loc[prim.delta.idxmax()]
print(f"Best single cell on ROC-AUC: {best_cell.model}/{best_cell.condition} "
      f"{best_cell.delta:+.5f} [{best_cell.lo:+.5f},{best_cell.hi:+.5f}]")

# Threshold-tuning control: does the untreated model reach the same F1?
piv = cv.groupby(["model", "condition"])[["f1", "f1_tuned", "accuracy",
                                          "bal_acc", "roc_auc", "pr_auc"]].mean()
print("\nF1: default-0.5 vs train-tuned threshold")
print(piv.round(4).to_string())

for m in MODELS:
    none_tuned = piv.loc[(m, "none"), "f1_tuned"]
    best_treat = piv.loc[m].drop("none")["f1_tuned"].max()
    best_name = piv.loc[m].drop("none")["f1_tuned"].idxmax()
    print(f"  {m}: untreated+tuned-threshold F1={none_tuned:.4f} vs "
          f"best treatment ({best_name}) F1={best_treat:.4f} "
          f"-> delta={best_treat - none_tuned:+.4f}")

summary_obj = dict(
    best_auc_condition=best_auc_cond,
    best_auc_delta=float(best_auc_val),
    class_weight_auc_delta=float(cw_mean),
)
print("\n", json.dumps(summary_obj, indent=2))
print("\nWrote cv_results.csv, paired_deltas.csv, holdout_results.csv")
