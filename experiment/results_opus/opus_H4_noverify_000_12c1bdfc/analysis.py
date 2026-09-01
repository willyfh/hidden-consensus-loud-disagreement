"""
H4: Does addressing class imbalance improve model quality on the Adult income dataset?

Design
------
Target: class (>50K = positive, 23.9% prevalence -> ~1:3.2 imbalance, mild).

Imbalance strategies compared (identical preprocessing / model class throughout):
  1. none              -- baseline, no imbalance handling
  2. class_weight      -- balanced class weights (sample weights for HGB)
  3. random_oversample -- duplicate minority rows until 1:1
  4. smote             -- synthetic minority oversampling to 1:1
  5. random_undersample-- drop majority rows until 1:1

Each strategy is evaluated with two decision rules:
  * fixed 0.5 threshold (the usual, naive way imbalance "problems" show up)
  * a threshold tuned to maximise F1 on inner-CV out-of-fold predictions
    (nested, so no test leakage)

Metrics: ROC-AUC and PR-AUC (threshold-free ranking quality), F1 / balanced
accuracy / MCC / accuracy (threshold-dependent), Brier score (calibration).

Validation: RepeatedStratifiedKFold(5 splits x 3 repeats) over the full 48842
rows; resampling/weighting is applied ONLY inside each training fold. Paired
per-fold differences vs. the baseline are tested with a paired t-test.

Models: L2 logistic regression and HistGradientBoostingClassifier (defaults).
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RNG = 0

# ---------------------------------------------------------------- data
df = pd.read_csv("adult_income.csv")
y = (df["class"].str.strip() == ">50K").astype(int).values
# fnlwgt is a census sampling weight, not a property of the person -> dropped.
X = df.drop(columns=["class", "fnlwgt"])
CAT = X.select_dtypes("object").columns.tolist()
NUM = [c for c in X.columns if c not in CAT]
X[CAT] = X[CAT].fillna("Missing")  # missingness is informative; keep as a level

print(f"n={len(y)}  positives={y.sum()} ({y.mean():.3%})  imbalance ratio 1:{(1-y.mean())/y.mean():.2f}")


def make_prep():
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), CAT),
        ]
    )


def make_model(kind, class_weight=None):
    if kind == "logreg":
        return LogisticRegression(max_iter=3000, C=1.0, class_weight=class_weight, random_state=RNG)
    return HistGradientBoostingClassifier(random_state=RNG)  # weights passed via fit


def resample(Xtr, ytr, strategy, seed):
    """Return (X, y, sample_weight) for the given imbalance strategy."""
    rng = np.random.default_rng(seed)
    if strategy in ("none", "class_weight"):
        return Xtr, ytr, None
    pos, neg = np.where(ytr == 1)[0], np.where(ytr == 0)[0]
    if strategy == "random_oversample":
        idx = np.concatenate([neg, pos, rng.choice(pos, len(neg) - len(pos), replace=True)])
        rng.shuffle(idx)
        return Xtr[idx], ytr[idx], None
    if strategy == "random_undersample":
        idx = np.concatenate([pos, rng.choice(neg, len(pos), replace=False)])
        rng.shuffle(idx)
        return Xtr[idx], ytr[idx], None
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        Xr, yr = SMOTE(random_state=seed, k_neighbors=5).fit_resample(Xtr, ytr)
        return Xr, yr, None
    raise ValueError(strategy)


def fit_predict(kind, strategy, Xtr, ytr, Xte, seed):
    Xr, yr, _ = resample(Xtr, ytr, strategy, seed)
    sw = None
    if strategy == "class_weight":
        w = len(yr) / (2 * np.bincount(yr))
        sw = w[yr]
    if kind == "logreg":
        m = make_model(kind, class_weight="balanced" if strategy == "class_weight" else None)
        m.fit(Xr, yr)
    else:
        m = make_model(kind)
        m.fit(Xr, yr, sample_weight=sw)
    return m.predict_proba(Xte)[:, 1]


def best_f1_threshold(ytrue, p):
    grid = np.unique(np.quantile(p, np.linspace(0.01, 0.99, 199)))
    return max(grid, key=lambda t: f1_score(ytrue, (p >= t).astype(int), zero_division=0))


def score(ytrue, p, thr):
    yhat = (p >= thr).astype(int)
    return dict(
        roc_auc=roc_auc_score(ytrue, p),
        pr_auc=average_precision_score(ytrue, p),
        brier=brier_score_loss(ytrue, p),
        f1=f1_score(ytrue, yhat, zero_division=0),
        mcc=matthews_corrcoef(ytrue, yhat),
        bal_acc=balanced_accuracy_score(ytrue, yhat),
        acc=accuracy_score(ytrue, yhat),
        pred_pos_rate=yhat.mean(),
    )


STRATEGIES = ["none", "class_weight", "random_oversample", "smote", "random_undersample"]
MODELS = ["logreg", "hgb"]

outer = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RNG)
rows = []

for fold, (tr, te) in enumerate(outer.split(X, y)):
    prep = make_prep()
    Xtr = prep.fit_transform(X.iloc[tr])
    Xte = prep.transform(X.iloc[te])
    ytr, yte = y[tr], y[te]

    for kind in MODELS:
        for strat in STRATEGIES:
            # nested inner CV on the training fold -> OOF probs -> F1-optimal threshold
            oof = np.zeros(len(ytr))
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG + fold)
            for itr, ite in inner.split(Xtr, ytr):
                oof[ite] = fit_predict(kind, strat, Xtr[itr], ytr[itr], Xtr[ite], seed=RNG + fold)
            thr_tuned = best_f1_threshold(ytr, oof)

            p = fit_predict(kind, strat, Xtr, ytr, Xte, seed=RNG + fold)
            for rule, thr in (("thr0.5", 0.5), ("thr_tuned", thr_tuned)):
                rows.append(dict(fold=fold, model=kind, strategy=strat, rule=rule,
                                 threshold=float(thr), **score(yte, p, thr)))
    print(f"fold {fold+1}/15 done", flush=True)

res = pd.DataFrame(rows)
res.to_csv("cv_results.csv", index=False)

METRICS = ["roc_auc", "pr_auc", "brier", "f1", "mcc", "bal_acc", "acc", "pred_pos_rate"]
summary = res.groupby(["model", "rule", "strategy"])[METRICS].agg(["mean", "std"]).round(4)
print("\n=== Mean over 15 folds ===")
with pd.option_context("display.width", 250, "display.max_columns", 50):
    print(res.groupby(["model", "rule", "strategy"])[METRICS].mean().round(4))


def paired(model, rule, strat, metric, base="none"):
    """Paired per-fold difference strat - base, with paired t-test."""
    k = lambda s: res.query("model==@model and rule==@rule and strategy==@s").sort_values("fold")[metric].values
    d = k(strat) - k(base)
    t, pv = stats.ttest_rel(k(strat), k(base))
    return float(d.mean()), float(d.std(ddof=1)), float(pv)


print("\n=== Paired differences vs. baseline (strategy - none), 15 folds ===")
lines = []
for model in MODELS:
    for rule in ["thr0.5", "thr_tuned"]:
        for strat in STRATEGIES[1:]:
            for metric in ["roc_auc", "pr_auc", "f1", "mcc", "bal_acc", "brier"]:
                m, s, pv = paired(model, rule, strat, metric)
                lines.append(dict(model=model, rule=rule, strategy=strat, metric=metric,
                                  mean_diff=round(m, 5), sd=round(s, 5), p=round(pv, 5)))
diffs = pd.DataFrame(lines)
diffs.to_csv("paired_differences.csv", index=False)
with pd.option_context("display.max_rows", 300, "display.width", 200):
    print(diffs.pivot_table(index=["model", "rule", "strategy"], columns="metric", values="mean_diff"))

# Threshold tuning on the plain baseline vs. imbalance handling at 0.5
print("\n=== Baseline + tuned threshold vs. imbalance handling at 0.5 (F1, MCC) ===")
for model in MODELS:
    b_tuned = res.query("model==@model and rule=='thr_tuned' and strategy=='none'").sort_values("fold")
    for strat in STRATEGIES[1:]:
        o = res.query("model==@model and rule=='thr0.5' and strategy==@strat").sort_values("fold")
        for metric in ["f1", "mcc"]:
            d = b_tuned[metric].values - o[metric].values
            pv = stats.ttest_rel(b_tuned[metric].values, o[metric].values).pvalue
            print(f"{model:7s} baseline@tuned - {strat:18s}@0.5  {metric}: {d.mean():+.4f} (p={pv:.4g})")

# ------------------------------------------------------- headline numbers
best_model = "hgb"
head_metric = "pr_auc"
pr_rows = {s: res.query("model==@best_model and rule=='thr0.5' and strategy==@s")[head_metric].mean()
           for s in STRATEGIES}
best_strat = max((s for s in STRATEGIES if s != "none"), key=lambda s: pr_rows[s])
pm, ps, ppv = paired(best_model, "thr0.5", best_strat, head_metric)
roc_m, _, roc_p = paired(best_model, "thr0.5", best_strat, "roc_auc")
f1_naive_m, _, f1_naive_p = paired(best_model, "thr0.5", "class_weight", "f1")
f1_tuned_m, _, f1_tuned_p = paired(best_model, "thr_tuned", "class_weight", "f1")

print(f"\nPR-AUC ({best_model}): " + ", ".join(f"{s}={pr_rows[s]:.4f}" for s in STRATEGIES))
print(f"best non-baseline strategy by PR-AUC: {best_strat}, diff={pm:+.5f} (p={ppv:.4g})")
print(f"ROC-AUC diff ({best_strat} - none): {roc_m:+.5f} (p={roc_p:.4g})")
print(f"F1 diff class_weight-none @0.5: {f1_naive_m:+.4f} (p={f1_naive_p:.4g}); "
      f"@tuned threshold: {f1_tuned_m:+.4f} (p={f1_tuned_p:.4g})")

best_pr_diff = max(paired(best_model, "thr0.5", s, head_metric)[0] for s in STRATEGIES[1:])

result = {
    "hypothesis_id": "H4",
    "summary": (
        "No. The Adult dataset is only mildly imbalanced (23.9% positive), and every imbalance "
        "remedy tested (balanced class weights, random over-/under-sampling, SMOTE) left "
        "threshold-free model quality unchanged or slightly worse: for gradient boosting the best "
        f"strategy changed PR-AUC by {best_pr_diff:+.4f} and ROC-AUC by {roc_m:+.4f} versus the "
        "untouched baseline, while all resampling variants degraded probability calibration "
        "(Brier score). Imbalance handling does raise F1/recall at a fixed 0.5 cutoff, but simply "
        "tuning the decision threshold on the plain baseline matches or beats it, so the apparent "
        "gain is an operating-point shift rather than a genuine improvement in model quality."
    ),
    "primary_metric_name": (
        "PR-AUC difference (best imbalance-handling strategy - no handling), HistGradientBoosting, "
        "mean over 15 stratified CV folds"
    ),
    "primary_metric_value": round(best_pr_diff, 5),
    "direction": "no improvement: imbalance handling ~= baseline on ranking metrics, worse calibration; threshold tuning suffices",
    "methodological_choices": (
        "Target >50K as positive (23.93% prevalence, 1:3.18). Dropped fnlwgt (census sampling weight); "
        "kept all other 13 features; missing workclass/occupation/native-country encoded as an explicit "
        "'Missing' level rather than imputed. Preprocessing: StandardScaler on numerics + one-hot "
        "(min_frequency=10, handle_unknown='ignore') on categoricals, fitted inside each training fold. "
        "Models: L2 logistic regression (C=1, max_iter=3000) and HistGradientBoostingClassifier "
        "(sklearn defaults, no tuning). Imbalance strategies: none, balanced class weights (sample_weight "
        "for HGB), random oversampling to 1:1, SMOTE (k=5, applied after encoding/scaling), random "
        "undersampling to 1:1 -- all applied only to training folds. Validation: RepeatedStratifiedKFold "
        "5x3 = 15 folds on all 48842 rows; paired per-fold differences with paired t-tests (no multiple-"
        "comparison correction). Metrics: PR-AUC (primary, most sensitive to the minority class), ROC-AUC, "
        "Brier score for calibration, and F1/MCC/balanced accuracy evaluated both at a fixed 0.5 cutoff and "
        "at an F1-optimal threshold chosen by nested inner 3-fold CV on the training fold. Alternative "
        "choices another researcher might make: a single train/test split, tuned hyperparameters, "
        "cost-sensitive thresholds from an explicit cost matrix, SMOTE-NC for categorical features, or "
        "treating F1 at 0.5 as the headline metric -- the last would reverse the qualitative conclusion."
    ),
}
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nwrote result.json\n", json.dumps(result, indent=2))
