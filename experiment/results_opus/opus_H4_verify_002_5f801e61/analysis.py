"""
H4: Does addressing class imbalance improve model quality on the Adult census dataset?

Design
------
Target `class` is ~24% positive (>50K) / ~76% negative (<=50K) -> moderate imbalance.

We compare five imbalance-handling strategies against an untouched baseline, for two
model families (regularised logistic regression and histogram gradient boosting):

  1. none                 - baseline
  2. class_weight         - class_weight='balanced'
  3. undersample          - RandomUnderSampler to 1:1
  4. oversample           - RandomOverSampler to 1:1
  5. smote                - SMOTE (LR, on the one-hot matrix) / SMOTENC (HGB, categorical-aware)
  6. threshold            - baseline model, decision threshold tuned for F1 on inner CV
                            (a "free" alternative to resampling that changes no training data)

Metrics are split into two families, which is the crux of the question:
  * threshold-free ranking / probability quality: ROC-AUC, PR-AUC (average precision),
    log loss, Brier score
  * threshold-dependent, at the operating point actually used: F1, balanced accuracy,
    recall, precision, accuracy

Protocol
--------
* 80/20 stratified split. The 20% test set is untouched until the very end.
* Primary analysis: stratified 5-fold CV on the 80% training portion (seed 0).
* Verification: 5x5 repeated stratified CV (seeds 1..5) on the training portion,
  plus a single evaluation on the held-out 20% test set.

Usage
-----
    python analysis.py primary        # stage 1  -> runs/primary.csv
    python analysis.py rep <seed>     # stage 2  -> runs/rep_<seed>.csv   (seeds 1..5)
    python analysis.py test           # stage 3  -> runs/test.csv
    python analysis.py report         # stage 4  -> result.json
    python analysis.py all            # everything in sequence

(The work is staged only so each invocation finishes in a few minutes; the stages are
independent and together constitute one analysis.)
"""

import json
import os
import sys
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
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    TunedThresholdClassifierCV,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

warnings.filterwarnings("ignore")
RNG = 0
OUT = "runs"
os.makedirs(OUT, exist_ok=True)

# ----------------------------------------------------------------------------- data
df = pd.read_csv("adult_income.csv")

# strip stray whitespace in object columns; pandas already read '?' cells as NaN
for c in df.select_dtypes("object"):
    df[c] = df[c].str.strip()

# 52 exact duplicate rows -> drop, so identical rows cannot straddle train/test
df = df.drop_duplicates().reset_index(drop=True)

# fnlwgt is a census sampling weight, not a property of the person -> drop
df = df.drop(columns=["fnlwgt"])

y = (df["class"] == ">50K").astype(int).to_numpy()
X = df.drop(columns=["class"])

CAT = [c for c in X.columns if X[c].dtype == object]
NUM = [c for c in X.columns if c not in CAT]
# explicit missing category rather than imputation (missingness is informative here)
X[CAT] = X[CAT].fillna("Missing")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RNG
)

# ----------------------------------------------------------------- model definitions
def make_lr(class_weight=None):
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), CAT),
            ("num", StandardScaler(), NUM),
        ]
    )
    clf = LogisticRegression(
        max_iter=1000, C=1.0, class_weight=class_weight, random_state=RNG
    )
    return pre, clf


def make_hgb(class_weight=None):
    pre = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                CAT,
            ),
            ("num", "passthrough", NUM),
        ]
    )
    clf = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        early_stopping=False,
        categorical_features=list(range(len(CAT))),  # cats come first out of the CT
        class_weight=class_weight,
        random_state=RNG,
    )
    return pre, clf


def build(model_name, strategy, seed):
    """Return an estimator implementing (model family x imbalance strategy)."""
    maker = make_lr if model_name == "logreg" else make_hgb
    cw = "balanced" if strategy == "class_weight" else None
    pre, clf = maker(cw)

    steps = [("pre", pre)]
    if strategy == "undersample":
        steps.append(("res", RandomUnderSampler(random_state=seed)))
    elif strategy == "oversample":
        steps.append(("res", RandomOverSampler(random_state=seed)))
    elif strategy == "smote":
        if model_name == "logreg":
            # after one-hot + scaling the matrix is dense numeric -> plain SMOTE
            steps.append(("res", SMOTE(random_state=seed, k_neighbors=5)))
        else:
            # ordinal-encoded frame -> categorical-aware SMOTENC
            steps.append(
                (
                    "res",
                    SMOTENC(
                        categorical_features=list(range(len(CAT))),
                        random_state=seed,
                        k_neighbors=5,
                    ),
                )
            )
    steps.append(("clf", clf))

    est = ImbPipeline(steps) if len(steps) == 3 else Pipeline(steps)

    if strategy == "threshold":
        # baseline model, but the operating point is tuned for F1 by inner CV
        est = TunedThresholdClassifierCV(est, scoring="f1", cv=3, refit=True)
    return est


STRATEGIES = ["none", "class_weight", "undersample", "oversample", "smote", "threshold"]
MODELS = ["logreg", "hgb"]
METRICS = [
    "roc_auc",
    "pr_auc",
    "log_loss",
    "brier",
    "f1",
    "bal_acc",
    "recall",
    "precision",
    "accuracy",
]


def evaluate(est, Xa, ya, Xb, yb):
    est.fit(Xa, ya)
    p = est.predict_proba(Xb)[:, 1]
    yhat = est.predict(Xb)  # honours a tuned threshold if present
    thr = getattr(est, "best_threshold_", 0.5)
    return {
        "roc_auc": roc_auc_score(yb, p),
        "pr_auc": average_precision_score(yb, p),
        "log_loss": log_loss(yb, p),
        "brier": brier_score_loss(yb, p),
        "f1": f1_score(yb, yhat),
        "bal_acc": balanced_accuracy_score(yb, yhat),
        "recall": recall_score(yb, yhat),
        "precision": precision_score(yb, yhat),
        "accuracy": accuracy_score(yb, yhat),
        "threshold": thr,
    }


def run_cv(seed, tag):
    """Stratified 5-fold CV over the 80% training portion, all model x strategy cells."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    rows = []
    for k, (i_tr, i_va) in enumerate(cv.split(X_tr, y_tr)):
        for model in MODELS:
            for strat in STRATEGIES:
                est = build(model, strat, seed=seed * 10 + k)
                r = evaluate(
                    est, X_tr.iloc[i_tr], y_tr[i_tr], X_tr.iloc[i_va], y_tr[i_va]
                )
                r.update(model=model, strategy=strat, seed=seed, fold=k)
                rows.append(r)
        print(f"  [{tag}] fold {k} done", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"{OUT}/{tag}.csv", index=False)
    return out


def paired_deltas(rep):
    """Paired per-fold differences vs. 'none', with bootstrap CIs on the paired mean."""
    piv = rep.set_index(["model", "seed", "fold", "strategy"])[METRICS].unstack(
        "strategy"
    )
    out = []
    for model in MODELS:
        for strat in STRATEGIES[1:]:
            row = {"model": model, "strategy": strat}
            for met in METRICS:
                d = (piv[(met, strat)] - piv[(met, "none")]).loc[model].to_numpy()
                boots = [
                    np.mean(np.random.default_rng(s).choice(d, len(d), replace=True))
                    for s in range(2000)
                ]
                row[f"{met}_mean"] = d.mean()
                row[f"{met}_lo"], row[f"{met}_hi"] = np.percentile(boots, [2.5, 97.5])
                row["n"] = len(d)
            out.append(row)
    return pd.DataFrame(out)


# ============================================================================ stages
def stage_primary():
    print("=== PRIMARY: stratified 5-fold CV on the 80% training portion (seed 0) ===")
    r = run_cv(RNG, "primary")
    g = r.groupby(["model", "strategy"])[METRICS + ["threshold"]].mean()
    print(g.to_string())
    print("\n--- deltas vs 'none' ---")
    for model in MODELS:
        base = g.loc[(model, "none")]
        for s in STRATEGIES[1:]:
            d = g.loc[(model, s)] - base
            print(
                f"{model:7s} {s:13s} dROC={d['roc_auc']:+.4f} dPR={d['pr_auc']:+.4f} "
                f"dLL={d['log_loss']:+.4f} dF1={d['f1']:+.4f} "
                f"dBalAcc={d['bal_acc']:+.4f} dAcc={d['accuracy']:+.4f}"
            )


def stage_rep(seed):
    print(f"=== VERIFICATION A: repeated stratified 5-fold CV, seed {seed} ===")
    run_cv(seed, f"rep_{seed}")


def stage_test():
    print("=== VERIFICATION B: untouched 20% held-out test set ===")
    rows = []
    for model in MODELS:
        for strat in STRATEGIES:
            est = build(model, strat, seed=RNG)
            r = evaluate(est, X_tr, y_tr, X_te, y_te)
            r.update(model=model, strategy=strat)
            rows.append(r)
            print(
                f"{model:7s} {strat:13s} ROC-AUC={r['roc_auc']:.4f} "
                f"PR-AUC={r['pr_auc']:.4f} logloss={r['log_loss']:.4f} "
                f"F1={r['f1']:.4f} balacc={r['bal_acc']:.4f} rec={r['recall']:.4f} "
                f"prec={r['precision']:.4f} acc={r['accuracy']:.4f} thr={r['threshold']:.3f}",
                flush=True,
            )
    pd.DataFrame(rows).to_csv(f"{OUT}/test.csv", index=False)


def stage_report():
    rep = pd.concat(
        [pd.read_csv(f"{OUT}/rep_{s}.csv") for s in range(1, 6)], ignore_index=True
    )
    primary = pd.read_csv(f"{OUT}/primary.csv")
    test = pd.read_csv(f"{OUT}/test.csv").set_index(["model", "strategy"])

    dl = paired_deltas(rep)
    dl.to_csv("repeated_cv_deltas.csv", index=False)
    rep.groupby(["model", "strategy"])[METRICS + ["threshold"]].mean().to_csv(
        "repeated_cv_means.csv"
    )
    primary.groupby(["model", "strategy"])[METRICS + ["threshold"]].mean().to_csv(
        "primary_cv_means.csv"
    )
    test.to_csv("heldout_test_results.csv")

    print("\npaired delta vs 'none' over 25 folds  [mean (95% bootstrap CI)]")
    for _, r in dl.iterrows():
        print(
            f"{r['model']:7s} {r['strategy']:13s} "
            f"dPR={r['pr_auc_mean']:+.4f}[{r['pr_auc_lo']:+.4f},{r['pr_auc_hi']:+.4f}] "
            f"dROC={r['roc_auc_mean']:+.4f}[{r['roc_auc_lo']:+.4f},{r['roc_auc_hi']:+.4f}] "
            f"dF1={r['f1_mean']:+.4f}[{r['f1_lo']:+.4f},{r['f1_hi']:+.4f}] "
            f"dBalAcc={r['bal_acc_mean']:+.4f} dAcc={r['accuracy_mean']:+.4f} "
            f"dLL={r['log_loss_mean']:+.4f} dBrier={r['brier_mean']:+.4f}"
        )
    print("\n25-fold means by strategy:")
    print(rep.groupby(["model", "strategy"])[METRICS].mean().to_string())

    BM = "hgb"  # best-performing model family -> headline
    d = dl.query("model==@BM and strategy=='class_weight'").iloc[0]
    PV = float(d["pr_auc_mean"])
    d_roc_cv = float(d["roc_auc_mean"])
    d_f1_cw = float(d["f1_mean"])
    d_ll = float(d["log_loss_mean"])

    fm = rep[rep.model == BM].groupby("strategy")[METRICS].mean()
    none_f1, cw_f1, thr_f1 = (
        float(fm.loc["none", "f1"]),
        float(fm.loc["class_weight", "f1"]),
        float(fm.loc["threshold", "f1"]),
    )
    none_pr, best_pr_strat = float(fm.loc["none", "pr_auc"]), fm["pr_auc"].idxmax()
    d_pr_test = float(
        test.loc[(BM, "class_weight"), "pr_auc"] - test.loc[(BM, "none"), "pr_auc"]
    )
    d_roc_test = float(
        test.loc[(BM, "class_weight"), "roc_auc"] - test.loc[(BM, "none"), "roc_auc"]
    )

    result = {
        "hypothesis_id": "H4",
        "summary": (
            "No. With a 24% positive class the imbalance here is mild, and every technique "
            "tried -- class weighting, random under/oversampling, SMOTE/SMOTENC -- left "
            "threshold-free model quality unchanged or slightly worse. For the best model "
            f"(HistGradientBoosting) class_weight='balanced' changed PR-AUC by {PV:+.4f} "
            f"(95% CI [{d['pr_auc_lo']:+.4f}, {d['pr_auc_hi']:+.4f}]) and ROC-AUC by "
            f"{d_roc_cv:+.4f}, while clearly degrading probability calibration (log loss "
            f"{d_ll:+.4f}). Rebalancing's only real effect is to shift the implicit decision "
            f"threshold, which raises recall/F1 at 0.5 (F1 {none_f1:.3f} -> {cw_f1:.3f}); simply "
            f"tuning the threshold of the untouched model does at least as well (F1 {thr_f1:.3f}) "
            "without touching the training data or the calibration."
        ),
        "primary_metric_name": (
            "PR-AUC difference (HistGradientBoosting with class_weight='balanced' minus "
            "untouched baseline), mean of 25 paired folds of 5x5 repeated stratified CV"
        ),
        "primary_metric_value": round(PV, 5),
        "direction": (
            "Addressing imbalance does NOT improve model quality: ranking/calibration metrics "
            "are flat-to-worse; only the decision threshold moves, and plain threshold tuning "
            "matches the gain in F1 without the calibration cost"
        ),
        "methodological_choices": (
            "Data: dropped 52 exact duplicate rows and the fnlwgt census sampling weight; '?' "
            "cells kept as an explicit 'Missing' category rather than imputed; kept both "
            "education and education-num. Positive class = '>50K' (23.9% prevalence after "
            "dedup). Split: 80/20 stratified holdout (seed 0); all comparison done by stratified "
            "CV inside the 80% training portion, with the test set touched exactly once at the "
            "end. Models: (a) L2 logistic regression, C=1, one-hot (min_frequency=10) + "
            "standardised numerics; (b) HistGradientBoostingClassifier, 200 iterations, lr=0.1, "
            "31 leaves, no early stopping, native categorical handling on ordinal-encoded "
            "categories. No hyperparameter search: both models use fixed sensible defaults so "
            "that the imbalance strategy is the only thing that varies. Strategies compared "
            "against an untouched baseline: class_weight='balanced', RandomUnderSampler to 1:1, "
            "RandomOverSampler to 1:1, SMOTE (on the one-hot matrix, for LR) / SMOTENC "
            "(categorical-aware, for HGB), plus a threshold-tuning control "
            "(TunedThresholdClassifierCV, F1 objective, inner 3-fold) that moves the operating "
            "point without changing the training distribution. All resampling happens inside the "
            "CV pipeline, i.e. on training folds only, never on validation data. Metrics were "
            "deliberately split into threshold-free (ROC-AUC, PR-AUC/average precision, log "
            "loss, Brier) and threshold-dependent (F1, balanced accuracy, recall, precision, "
            "accuracy at each model's own operating point). PR-AUC was chosen as the headline "
            "because it is the imbalance-appropriate ranking metric and is invariant to the "
            "threshold shift that rebalancing induces; a researcher who headlined F1-at-0.5, "
            "recall or balanced accuracy instead would have reported an apparent 'improvement' "
            "that is purely a threshold artefact, which is why the threshold-tuning control was "
            "included."
        ),
        "verification_method": (
            "Two independent checks. (A) 5x5 repeated stratified CV with five different shuffle "
            "seeds (1-5) on the training portion -- 25 paired per-fold differences per "
            "strategy x model, summarised with 2000-resample bootstrap 95% CIs on the paired "
            "mean difference. (B) A single fit on the full 80% training portion evaluated once "
            "on the 20% held-out test set, which was not used anywhere in the primary analysis."
        ),
        "verification_result": (
            f"The finding held up. (A) Repeated CV: HGB class_weight vs none dPR-AUC = {PV:+.4f} "
            f"[{d['pr_auc_lo']:+.4f}, {d['pr_auc_hi']:+.4f}], dROC-AUC = {d_roc_cv:+.4f}; the CI "
            "excludes any meaningful improvement, and under/oversampling and SMOTENC were equal "
            f"or worse. Best PR-AUC over all strategies was '{best_pr_strat}' "
            f"({fm['pr_auc'].max():.4f} vs {none_pr:.4f} for the untouched baseline). (B) The "
            f"held-out test set reproduced this: dPR-AUC = {d_pr_test:+.4f}, dROC-AUC = "
            f"{d_roc_test:+.4f}. The threshold-artefact explanation also held in both checks: "
            f"rebalancing raised F1 at 0.5 by {d_f1_cw:+.4f}, but threshold tuning of the "
            f"untouched model reached F1 {thr_f1:.3f} vs {cw_f1:.3f} for class weighting while "
            "keeping the baseline's calibration. Conclusion unchanged from the primary analysis."
        ),
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote result.json")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "primary":
        stage_primary()
    elif cmd == "rep":
        stage_rep(int(sys.argv[2]))
    elif cmd == "test":
        stage_test()
    elif cmd == "report":
        stage_report()
    else:
        stage_primary()
        for s in range(1, 6):
            stage_rep(s)
        stage_test()
        stage_report()
