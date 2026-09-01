"""
H1: Does the choice of model family meaningfully affect predictive performance
on the UCI Adult (Census Income) dataset?

Design
------
* Data: adult_income.csv (48,842 rows). Target `class` binarised to >50K = 1 (23.9% positive).
* Exact duplicate rows dropped (52) to avoid train/test leakage.
* `fnlwgt` dropped: it is a census sampling design weight, not an attribute of the
  individual, so it should not carry signal about that person's income.
* Missing values (workclass / occupation / native-country, originally '?') are kept as an
  explicit "Missing" category rather than being dropped or imputed.
* Split: stratified 80% analysis set / 20% held-out test set. The test set is untouched
  until the verification stage.
* Model families compared (each given a small, fair hyperparameter search by inner 3-fold
  CV on the analysis set, on the same preprocessed features):
      - Dummy (stratified prior)          -- floor
      - Logistic regression (L2)          -- linear
      - Gaussian naive Bayes              -- generative
      - k-nearest neighbours              -- instance-based
      - Decision tree (CART)              -- single tree
      - Random forest                     -- bagged trees
      - Hist. gradient boosting           -- boosted trees
      - MLP (1 hidden layer)              -- neural net
* Preprocessing is family-appropriate but the *information* is identical: one-hot +
  standardised numerics for the scale/distance-sensitive learners (logreg, NB, kNN, MLP),
  ordinal codes for the tree learners.
* Primary metric: ROC-AUC (threshold-free, robust to the 3:1 class imbalance).
  Average precision, accuracy and balanced accuracy are reported alongside.
* No class re-weighting: ROC-AUC/AP are ranking metrics, and all families see the
  same class prior.

Verification
------------
1. 5x5 repeated stratified CV (5 different seeds) on the analysis set -> mean +- sd of the
   per-fold paired differences between families.
2. Refit on the full analysis set, score the untouched 20% held-out test set, and bootstrap
   (2000 paired resamples) a 95% CI on the ROC-AUC difference.

Run as:  python analysis.py tune | main | repcv | heldout | report
(staged so that each stage completes in the foreground; stages cache to CACHE_DIR).
"""

import json
import os
import pickle
import sys
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
                             balanced_accuracy_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RNG = 20260831

# ---------------------------------------------------------------- data
df = pd.read_csv("adult_income.csv")
df = df.drop_duplicates().reset_index(drop=True)
y = (df["class"].str.strip() == ">50K").astype(int).values
X = df.drop(columns=["class", "fnlwgt"])

NUM = [c for c in X.columns if X[c].dtype != object]
CAT = [c for c in X.columns if X[c].dtype == object]
X[CAT] = X[CAT].fillna("Missing")
print(f"rows={len(X)}  positives={y.mean():.4f}  num={NUM}  cat={CAT}")

X_ana, X_test, y_ana, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RNG
)
print(f"analysis={X_ana.shape}  heldout={X_test.shape}")

# ------------------------------------------------------- preprocessors
def onehot_pre():
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                              sparse_output=False), CAT),
    ])


def ordinal_pre():
    return ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT),
    ])


CAT_IDX = [len(NUM) + i for i in range(len(CAT))]

# ------------------------------------------------ families + small grids
FAMILIES = {
    "Dummy": (
        Pipeline([("pre", ordinal_pre()),
                  ("clf", DummyClassifier(strategy="stratified", random_state=RNG))]),
        {},
    ),
    "LogisticRegression": (
        Pipeline([("pre", onehot_pre()),
                  ("clf", LogisticRegression(max_iter=3000, solver="lbfgs"))]),
        {"clf__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    ),
    "GaussianNB": (
        Pipeline([("pre", onehot_pre()), ("clf", GaussianNB())]),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]},
    ),
    "kNN": (
        Pipeline([("pre", onehot_pre()),
                  ("clf", KNeighborsClassifier(n_jobs=-1))]),
        {"clf__n_neighbors": [25, 50, 100], "clf__weights": ["uniform", "distance"]},
    ),
    "DecisionTree": (
        Pipeline([("pre", ordinal_pre()),
                  ("clf", DecisionTreeClassifier(random_state=RNG))]),
        {"clf__max_depth": [6, 10, 14, None], "clf__min_samples_leaf": [1, 20, 100]},
    ),
    "RandomForest": (
        Pipeline([("pre", ordinal_pre()),
                  ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                                 random_state=RNG))]),
        {"clf__min_samples_leaf": [1, 5, 20], "clf__max_features": ["sqrt", 0.5]},
    ),
    "HistGradientBoosting": (
        Pipeline([("pre", ordinal_pre()),
                  ("clf", HistGradientBoostingClassifier(
                      categorical_features=CAT_IDX, early_stopping=True,
                      validation_fraction=0.1, random_state=RNG))]),
        {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31, 63],
         "clf__l2_regularization": [0.0, 1.0]},
    ),
    "MLP": (
        Pipeline([("pre", onehot_pre()),
                  ("clf", MLPClassifier(max_iter=300, early_stopping=True,
                                        n_iter_no_change=10, random_state=RNG))]),
        {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-2]},
    ),
}

# ------------------------------------------------------------ stage plumbing
CACHE_DIR = os.environ.get("H1_CACHE", ".h1_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def cpath(name):
    return os.path.join(CACHE_DIR, name + ".pkl")


def save(name, obj):
    with open(cpath(name), "wb") as f:
        pickle.dump(obj, f)


def load(name):
    with open(cpath(name), "rb") as f:
        return pickle.load(f)


def tuned_pipelines():
    """Rebuild each family's pipeline with the hyperparameters chosen in `tune`."""
    params = load("best_params")
    out = {}
    for name, (pipe, _) in FAMILIES.items():
        out[name] = pipe.set_params(**params[name]) if params[name] else pipe
    return out


# ------------------------------------------------------- stage 1: tune
def stage_tune():
    inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=RNG)
    params, inner_scores = {}, {}
    print("=== hyperparameter search (inner 3-fold CV on analysis set, ROC-AUC) ===")
    for name, (pipe, grid) in FAMILIES.items():
        t0 = time.time()
        if grid:
            gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner, n_jobs=-1,
                              refit=False)
            gs.fit(X_ana, y_ana)
            params[name], inner_scores[name] = gs.best_params_, gs.best_score_
            print(f"{name:22s} inner AUC={gs.best_score_:.4f}  {gs.best_params_}"
                  f"  [{time.time()-t0:.0f}s]")
        else:
            params[name], inner_scores[name] = {}, float("nan")
            print(f"{name:22s} (no grid)")
    save("best_params", params)
    save("inner_scores", inner_scores)


# ------------------------------ stage 2: main estimate, 5-fold CV on analysis set
def eval_cv(pipe, X, y, cv):
    rows = []
    for k, (tr, te) in enumerate(cv.split(X, y)):
        p = pipe.fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        yhat = (p >= 0.5).astype(int)
        rows.append({"fold": k,
                     "roc_auc": roc_auc_score(y[te], p),
                     "avg_prec": average_precision_score(y[te], p),
                     "accuracy": accuracy_score(y[te], yhat),
                     "bal_acc": balanced_accuracy_score(y[te], yhat)})
    return pd.DataFrame(rows)


def stage_main():
    best = tuned_pipelines()
    main_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
    print("=== main estimate: stratified 5-fold CV on the analysis set ===")
    main = {}
    for name, pipe in best.items():
        t0 = time.time()
        main[name] = eval_cv(pipe, X_ana, y_ana, main_cv)
        m = main[name].mean(numeric_only=True)
        print(f"{name:22s} AUC={m.roc_auc:.4f}  AP={m.avg_prec:.4f}  "
              f"acc={m.accuracy:.4f}  bal_acc={m.bal_acc:.4f}  [{time.time()-t0:.0f}s]")
    main_tab = (pd.DataFrame({n: d.mean(numeric_only=True) for n, d in main.items()})
                .T.drop(columns="fold").sort_values("roc_auc", ascending=False))
    save("main_tab", main_tab)
    print("\n" + main_tab.to_string())


# ------------------- stage 3a (verification 1): 5x5 repeated stratified CV
def stage_repcv(only=None):
    """Per-family caching so the stage can be resumed family by family."""
    best = tuned_pipelines()
    rep_cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=777)
    splits = list(rep_cv.split(X_ana, y_ana))
    print("=== verification 1: 5x5 repeated stratified CV (5 seeds), analysis set ===")
    for name, pipe in best.items():
        if only and name not in only:
            continue
        if os.path.exists(cpath("rep_" + name)):
            print(f"{name:22s} cached")
            continue
        t0 = time.time()
        scores = []
        for tr, te in splits:
            p = pipe.fit(X_ana.iloc[tr], y_ana[tr]).predict_proba(X_ana.iloc[te])[:, 1]
            scores.append(roc_auc_score(y_ana[te], p))
        save("rep_" + name, np.array(scores))
        print(f"{name:22s} AUC={np.mean(scores):.4f} +- {np.std(scores):.4f} "
              f"[{np.min(scores):.4f},{np.max(scores):.4f}]  [{time.time()-t0:.0f}s]")


# ------------- stage 3b (verification 2): untouched held-out test predictions
def stage_heldout(only=None):
    best = tuned_pipelines()
    print("=== verification 2: fit on full analysis set, predict untouched held-out 20% ===")
    for name, pipe in best.items():
        if only and name not in only:
            continue
        if os.path.exists(cpath("test_" + name)):
            print(f"{name:22s} cached")
            continue
        t0 = time.time()
        p = pipe.fit(X_ana, y_ana).predict_proba(X_test)[:, 1]
        save("test_" + name, p)
        print(f"{name:22s} test AUC={roc_auc_score(y_test, p):.4f}  [{time.time()-t0:.0f}s]")


# ------------------------------------------ stage 4: aggregate -> result.json
def stage_report():
    main_tab = load("main_tab")
    rep_df = pd.DataFrame({n: load("rep_" + n) for n in FAMILIES})
    test_p = {n: load("test_" + n) for n in FAMILIES}

    real = [n for n in main_tab.index if n != "Dummy"]
    spread = main_tab.loc[real, "roc_auc"].max() - main_tab.loc[real, "roc_auc"].min()
    print("main 5-fold CV (analysis set):\n" + main_tab.to_string())
    print(f"\nCV spread across non-trivial families = {spread:.4f}")

    # --- repeated CV
    rep_mean = rep_df.mean().sort_values(ascending=False)
    print("\nrepeated 5x5 CV mean ROC-AUC:\n" +
          pd.DataFrame({"mean": rep_mean, "sd": rep_df.std()}).to_string())
    top = rep_mean.index[0]
    print(f"\npaired per-fold differences vs {top} (25 folds):")
    paired = {}
    for name in rep_mean.index[1:]:
        d = rep_df[top] - rep_df[name]
        paired[name] = (float(d.mean()), float(d.std()), float((d > 0).mean()))
        print(f"  {top} - {name:22s}: {d.mean():+.4f} +- {d.std():.4f}  "
              f"wins {(d>0).mean()*100:.0f}% of folds")
    rank_stability = (rep_df.drop(columns="Dummy").rank(axis=1, ascending=False)
                      .agg(["mean", "std"]).T)
    print("\nper-fold rank of each family (1 = best):\n" + rank_stability.to_string())

    # --- held-out test
    rows = []
    for name, p in test_p.items():
        yhat = (p >= 0.5).astype(int)
        rows.append({"model": name,
                     "roc_auc": roc_auc_score(y_test, p),
                     "avg_prec": average_precision_score(y_test, p),
                     "accuracy": accuracy_score(y_test, yhat),
                     "bal_acc": balanced_accuracy_score(y_test, yhat)})
    test_tab = (pd.DataFrame(rows).set_index("model")
                .sort_values("roc_auc", ascending=False))
    print("\nheld-out test set:\n" + test_tab.to_string())

    rng = np.random.default_rng(RNG)
    n = len(y_test)
    idx = rng.integers(0, n, size=(2000, n))

    def boot_ci(pa, pb):
        d = np.empty(len(idx))
        for i, ii in enumerate(idx):
            yy = y_test[ii]
            d[i] = roc_auc_score(yy, pa[ii]) - roc_auc_score(yy, pb[ii])
        return float(d.mean()), np.percentile(d, [2.5, 97.5])

    test_top = test_tab.index[0]
    print(f"\nbootstrap 95% CI on held-out ROC-AUC difference vs {test_top} "
          f"(2000 paired resamples):")
    boots = {}
    for name in test_tab.index[1:]:
        m, ci = boot_ci(test_p[test_top], test_p[name])
        boots[name] = (m, ci)
        flag = "" if ci[0] > 0 else "   <-- CI includes 0"
        print(f"  {test_top} - {name:22s}: {m:+.4f}  95% CI "
              f"[{ci[0]:+.4f}, {ci[1]:+.4f}]{flag}")

    test_real = [i for i in test_tab.index if i != "Dummy"]
    test_spread = float(test_tab.loc[test_real, "roc_auc"].max() -
                        test_tab.loc[test_real, "roc_auc"].min())
    test_gap = float(test_tab.loc["HistGradientBoosting", "roc_auc"] -
                     test_tab.loc["LogisticRegression", "roc_auc"])
    gap_m, gap_ci = boot_ci(test_p["HistGradientBoosting"],
                            test_p["LogisticRegression"])
    d_rep = rep_df["HistGradientBoosting"] - rep_df["LogisticRegression"]
    print(f"\nheld-out spread (non-trivial families) = {test_spread:.4f}")
    print(f"held-out HGB - LogReg = {test_gap:+.4f}  95% CI "
          f"[{gap_ci[0]:+.4f}, {gap_ci[1]:+.4f}]")
    print(f"repeated-CV HGB - LogReg = {d_rep.mean():+.4f} +- {d_rep.std():.4f} "
          f"(wins {(d_rep>0).mean()*100:.0f}% of 25 folds)")

    # ----------------------------------------------------------- result.json
    STRONG = ["HistGradientBoosting", "RandomForest", "LogisticRegression", "MLP"]
    strong_spread = float(test_tab.loc[STRONG, "roc_auc"].max() -
                          test_tab.loc[STRONG, "roc_auc"].min())
    summary = (
        f"Yes, model family matters, but the effect is tiered rather than uniform. Across "
        f"eight families tuned on identical features, held-out ROC-AUC spans "
        f"{test_tab.loc[test_real,'roc_auc'].min():.3f}-"
        f"{test_tab.loc[test_real,'roc_auc'].max():.3f}; gradient boosting is the best family "
        f"and beats regularised logistic regression by {test_gap:+.4f} AUC (bootstrap 95% CI "
        f"[{gap_ci[0]:+.3f}, {gap_ci[1]:+.3f}]) - a small but statistically reliable and "
        f"reproducible margin. The strong families (boosting, random forest, logistic "
        f"regression, MLP) sit within {strong_spread:.3f} AUC of one another while naive Bayes "
        f"and a single tree fall clearly behind, so family choice matters mainly for avoiding "
        f"weak families rather than for the last point of AUC among strong ones."
    )
    result = {
        "hypothesis_id": "H1",
        "summary": summary,
        "primary_metric_name": ("ROC-AUC difference (HistGradientBoosting - "
                                "LogisticRegression) on held-out test set"),
        "primary_metric_value": round(test_gap, 4),
        "direction": (f"{' > '.join(list(test_tab.index[:4]))}; gradient boosting best, "
                      f"its margin over logistic regression small (~{abs(test_gap):.3f} AUC) "
                      f"but consistent; GaussianNB/DecisionTree clearly worse"),
        "methodological_choices": (
            "Target binarised >50K=1 (23.9% positive). Dropped 52 exact duplicate rows to avoid "
            "train/test leakage, and dropped `fnlwgt` (a census sampling design weight, not an "
            "attribute of the individual). Missing workclass/occupation/native-country kept as an "
            "explicit 'Missing' level rather than imputed or dropped; `education` kept alongside "
            "its ordinal twin `education-num`. Stratified 80/20 analysis/held-out split "
            "(seed 20260831). Eight families compared: Dummy(stratified), L2 logistic regression, "
            "GaussianNB, kNN, CART, random forest (400 trees), HistGradientBoosting (native "
            "categorical handling, early stopping) and a 1-hidden-layer MLP. Each family got a "
            "small grid search by inner 3-fold CV on the analysis set scored by ROC-AUC, so no "
            "family is penalised by poor defaults. Encoding is family-appropriate but carries "
            "identical information: one-hot (min_frequency=10) + standardised numerics for "
            "logreg/NB/kNN/MLP, ordinal codes for the tree learners. Primary metric ROC-AUC "
            "(threshold-free, insensitive to the ~3:1 imbalance); average precision, accuracy and "
            "balanced accuracy at a 0.5 threshold reported alongside. No class weighting or "
            "resampling. Another researcher might have kept fnlwgt or used it as a sample weight, "
            "chosen accuracy/F1 as the headline metric, one-hot-encoded for the trees too, left "
            "hyperparameters at defaults, used the canonical 32561/16281 train/test split, or "
            "added a gradient-boosting library (XGBoost/LightGBM were unavailable here)."
        ),
        "verification_method": (
            "Two independent checks. (1) 5x5 repeated stratified CV (25 folds, 5 different seeds, "
            "seed 777) on the analysis set, with all families scored on identical folds so per-fold "
            "differences are paired; per-fold ranks tracked for ordering stability. (2) Refit on the "
            "full analysis set and scored the 20% held-out test set that was untouched during "
            "tuning and CV, with a 2000-resample paired bootstrap 95% CI on ROC-AUC differences."
        ),
        "verification_result": (
            f"Held up. In 5x5 repeated CV HistGradientBoosting had the highest mean ROC-AUC "
            f"({rep_mean['HistGradientBoosting']:.4f} +- "
            f"{rep_df['HistGradientBoosting'].std():.4f}) and beat logistic regression on "
            f"{(d_rep>0).mean()*100:.0f}% of the 25 paired folds (mean gap {d_rep.mean():+.4f} "
            f"+- {d_rep.std():.4f}); the family ordering was stable across seeds (mean per-fold "
            f"rank of HGB = {rank_stability.loc['HistGradientBoosting','mean']:.2f}). On the "
            f"untouched held-out 20% the gap was {test_gap:+.4f} with a bootstrap 95% CI of "
            f"[{gap_ci[0]:+.4f}, {gap_ci[1]:+.4f}], excluding zero. Revised estimate for the "
            f"headline difference: {min(abs(d_rep.mean()), abs(test_gap)):.3f}-"
            f"{max(abs(d_rep.mean()), abs(test_gap)):.3f} ROC-AUC."
        ),
        "_detail": {
            "main_5fold_cv_analysis_set": main_tab.round(4).to_dict(orient="index"),
            "repeated_cv_5x5_roc_auc_mean": rep_mean.round(4).to_dict(),
            "repeated_cv_5x5_roc_auc_sd": rep_df.std().round(4).to_dict(),
            "repeated_cv_mean_rank": rank_stability["mean"].round(2).to_dict(),
            "repeated_cv_paired_diff_vs_best": {k: [round(v[0], 4), round(v[1], 4),
                                                    round(v[2], 3)]
                                                for k, v in paired.items()},
            "heldout_test": test_tab.round(4).to_dict(orient="index"),
            "heldout_bootstrap_ci_vs_best": {k: [round(v[0], 4), round(v[1][0], 4),
                                                 round(v[1][1], 4)]
                                             for k, v in boots.items()},
            "heldout_spread_nontrivial_families_auc": round(test_spread, 4),
            "heldout_spread_strong_families_auc": round(strong_spread, 4),
            "cv_spread_nontrivial_families_auc": round(float(spread), 4),
            "repeated_cv_HGB_minus_LogReg": [round(float(d_rep.mean()), 4),
                                             round(float(d_rep.std()), 4),
                                             round(float((d_rep > 0).mean()), 3)],
            "tuned_params": load("best_params"),
        },
    }
    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\nwrote result.json")
    print(json.dumps({k: v for k, v in result.items() if k != "_detail"}, indent=2))


if __name__ == "__main__":
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    only = sys.argv[2:] or None
    if stage in ("tune", "all"):
        stage_tune()
    if stage in ("main", "all"):
        stage_main()
    if stage in ("repcv", "all"):
        stage_repcv(only)
    if stage in ("heldout", "all"):
        stage_heldout(only)
    if stage in ("report", "all"):
        stage_report()
