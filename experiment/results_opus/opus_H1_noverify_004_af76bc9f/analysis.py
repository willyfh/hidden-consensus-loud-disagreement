"""
H1: Does the choice of model family meaningfully affect predictive performance
    on the UCI Adult (Census Income) dataset?

Design
------
* One shared preprocessing contract per model "flavour" (dense one-hot + scaling
  for distance/gradient-based learners; ordinal + native categorical handling for
  the histogram gradient booster), so that each family is given the representation
  it is normally given in practice.
* 80/20 stratified train/test split (seed 0). All tuning happens inside the
  training set only (3-fold stratified CV, small per-family grid).
* Generalisation is estimated two ways:
    1. 5-fold stratified CV on the *training* set with the tuned pipeline
       -> mean +/- sd ROC-AUC, gives a sense of run-to-run noise.
    2. Refit on the full training set, scored once on the held-out test set,
       with a paired bootstrap (2000 resamples) over test rows for CIs on
       between-family differences.
* Primary metric: ROC-AUC (threshold-free, insensitive to the 24/76 class
  imbalance). Secondary: average precision, accuracy & F1 at 0.5, Brier score.

Results for each model are cached to _cache/<name>.json so the script can be
re-run incrementally.
"""

import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE = os.path.join(HERE, "_cache")
os.makedirs(CACHE, exist_ok=True)
SEED = 0


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load():
    df = pd.read_csv(os.path.join(HERE, "adult_income.csv"))
    # 52 exact duplicate rows: dropped so that identical records cannot straddle
    # the train/test boundary and leak.
    df = df.drop_duplicates().reset_index(drop=True)
    # 'education' is a string re-encoding of the ordinal 'education-num';
    # keeping both is pure redundancy, so the string column is dropped.
    df = df.drop(columns=["education"])
    y = (df.pop("class").str.strip() == ">50K").astype(int).to_numpy()
    return df, y


NUM = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = ["workclass", "marital-status", "occupation", "relationship", "race", "sex",
       "native-country"]


def prep_dense():
    """One-hot + standardised numerics: for linear / kNN / MLP / NB / SVM / trees."""
    return ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                fill_value="__missing__")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               min_frequency=10,
                                               sparse_output=False))]), CAT),
    ])


def prep_native():
    """Ordinal-coded categoricals for HistGradientBoosting's native support."""
    return ColumnTransformer([
        ("num", SimpleImputer(strategy="median"), NUM),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="constant",
                                                fill_value="__missing__")),
                          ("od", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                unknown_value=-1))]), CAT),
    ])


# --------------------------------------------------------------------------- #
# Model families + small per-family grids (tuned on train only)
# --------------------------------------------------------------------------- #
def catalogue():
    dense = prep_dense
    m = {}

    m["dummy_prior"] = (
        Pipeline([("prep", dense()), ("clf", DummyClassifier(strategy="prior"))]), {})

    m["logreg_l2"] = (
        Pipeline([("prep", dense()),
                  ("clf", LogisticRegression(max_iter=3000, random_state=SEED))]),
        {"clf__C": [0.03, 0.3, 1.0, 10.0]})

    m["linear_svm"] = (
        Pipeline([("prep", dense()),
                  ("clf", CalibratedClassifierCV(
                      LinearSVC(dual="auto", random_state=SEED, max_iter=5000),
                      method="sigmoid", cv=3))]),
        {"clf__estimator__C": [0.03, 0.3, 1.0]})

    m["gaussian_nb"] = (
        Pipeline([("prep", dense()), ("clf", GaussianNB())]),
        {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]})

    m["knn"] = (
        Pipeline([("prep", dense()),
                  ("clf", KNeighborsClassifier(n_jobs=-1))]),
        {"clf__n_neighbors": [25, 75], "clf__weights": ["uniform", "distance"]})

    m["decision_tree"] = (
        Pipeline([("prep", dense()),
                  ("clf", DecisionTreeClassifier(random_state=SEED))]),
        {"clf__max_depth": [6, 12, None], "clf__min_samples_leaf": [1, 20, 100]})

    m["random_forest"] = (
        Pipeline([("prep", dense()),
                  ("clf", RandomForestClassifier(n_estimators=400, n_jobs=-1,
                                                 random_state=SEED))]),
        {"clf__min_samples_leaf": [1, 5, 20], "clf__max_features": ["sqrt", 0.3]})

    m["mlp"] = (
        Pipeline([("prep", dense()),
                  ("clf", MLPClassifier(random_state=SEED, max_iter=100,
                                        early_stopping=True, n_iter_no_change=8))]),
        {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-2]})

    cat_mask = [False] * len(NUM) + [True] * len(CAT)
    m["hist_gbm"] = (
        Pipeline([("prep", prep_native()),
                  ("clf", HistGradientBoostingClassifier(
                      categorical_features=cat_mask, max_iter=500,
                      early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=20, random_state=SEED))]),
        {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [31, 63],
         "clf__min_samples_leaf": [20, 50]})

    return m


# --------------------------------------------------------------------------- #
# Fit / evaluate one family
# --------------------------------------------------------------------------- #
def scores(y, p, thr=0.5):
    yh = (p >= thr).astype(int)
    return {
        "roc_auc": roc_auc_score(y, p),
        "average_precision": average_precision_score(y, p),
        "accuracy": accuracy_score(y, yh),
        "f1": f1_score(y, yh),
        "brier": brier_score_loss(y, p),
        "log_loss": log_loss(y, np.clip(p, 1e-9, 1 - 1e-9)),
    }


def run_one(name, Xtr, ytr, Xte, yte):
    out = os.path.join(CACHE, f"{name}.json")
    if os.path.exists(out):
        return json.load(open(out))

    est, grid = catalogue()[name]
    t0 = time.time()

    if grid:
        gs = GridSearchCV(est, grid, scoring="roc_auc",
                          cv=StratifiedKFold(3, shuffle=True, random_state=SEED),
                          n_jobs=-1, refit=True)
        gs.fit(Xtr, ytr)
        best, params = gs.best_estimator_, {k: str(v) for k, v in gs.best_params_.items()}
    else:
        best = clone(est).fit(Xtr, ytr)
        params = {}

    # honest CV spread on the training set with the selected configuration
    cv = cross_val_score(clone(best), Xtr, ytr, scoring="roc_auc", n_jobs=-1,
                         cv=StratifiedKFold(5, shuffle=True, random_state=SEED))

    p_te = best.predict_proba(Xte)[:, 1]
    res = {
        "name": name,
        "best_params": params,
        "cv_auc_mean": float(cv.mean()),
        "cv_auc_sd": float(cv.std(ddof=1)),
        "test": scores(yte, p_te),
        "fit_seconds": time.time() - t0,
    }
    np.save(os.path.join(CACHE, f"{name}_pte.npy"), p_te)
    json.dump(res, open(out, "w"), indent=2)
    print(f"{name:16s} cv={cv.mean():.4f}+/-{cv.std(ddof=1):.4f} "
          f"test_auc={res['test']['roc_auc']:.4f} ({res['fit_seconds']:.0f}s)", flush=True)
    return res


def main():
    from sklearn.model_selection import train_test_split

    X, y = load()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                          random_state=SEED)
    wanted = sys.argv[1:] or list(catalogue())
    for n in wanted:
        run_one(n, Xtr, ytr, Xte, yte)


if __name__ == "__main__":
    main()


# --------------------------------------------------------------------------- #
# Report: paired bootstrap over held-out test rows
# --------------------------------------------------------------------------- #
def report():
    from sklearn.model_selection import train_test_split

    X, y = load()
    _, _, _, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)

    names = [n for n in catalogue()]
    res = {n: json.load(open(os.path.join(CACHE, f"{n}.json"))) for n in names}
    P = {n: np.load(os.path.join(CACHE, f"{n}_pte.npy")) for n in names}

    order = sorted(names, key=lambda n: -res[n]["test"]["roc_auc"])
    print(f"\n{'family':16s} {'cv_auc':>16s} {'test_auc':>9s} {'test_AP':>8s} "
          f"{'acc':>7s} {'f1':>7s} {'brier':>7s}")
    for n in order:
        t = res[n]["test"]
        print(f"{n:16s} {res[n]['cv_auc_mean']:.4f}+/-{res[n]['cv_auc_sd']:.4f} "
              f"{t['roc_auc']:9.4f} {t['average_precision']:8.4f} "
              f"{t['accuracy']:7.4f} {t['f1']:7.4f} {t['brier']:7.4f}")
    print("\nselected hyperparameters:")
    for n in order:
        print(f"  {n:16s} {res[n]['best_params']}")

    # paired bootstrap: resample test rows, recompute every model's AUC on the
    # same resample, so differences are paired and share sampling noise.
    rng = np.random.default_rng(SEED)
    B = 2000
    boot = {n: np.empty(B) for n in names}
    idx_pos = np.where(yte == 1)[0]
    idx_neg = np.where(yte == 0)[0]
    for b in range(B):
        # stratified bootstrap keeps the positive rate fixed across resamples
        i = np.concatenate([rng.choice(idx_pos, idx_pos.size, replace=True),
                            rng.choice(idx_neg, idx_neg.size, replace=True)])
        yb = yte[i]
        for n in names:
            boot[n][b] = roc_auc_score(yb, P[n][i])

    def ci(d):
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

    pairs = [("hist_gbm", "logreg_l2"), ("hist_gbm", "random_forest"),
             ("hist_gbm", "mlp"), ("random_forest", "logreg_l2"),
             ("mlp", "logreg_l2"), ("logreg_l2", "linear_svm"),
             ("logreg_l2", "decision_tree"), ("logreg_l2", "knn"),
             ("hist_gbm", "gaussian_nb")]
    print(f"\npaired bootstrap ROC-AUC differences ({B} resamples):")
    diffs = {}
    for a, b_ in pairs:
        d = boot[a] - boot[b_]
        lo, hi = ci(d)
        p_gt = float((d <= 0).mean())
        diffs[f"{a}-{b_}"] = {"delta": float(res[a]['test']['roc_auc']
                                             - res[b_]['test']['roc_auc']),
                              "ci95": [lo, hi], "p_le_0": p_gt}
        print(f"  {a:14s} - {b_:14s} = {diffs[f'{a}-{b_}']['delta']:+.4f} "
              f"[{lo:+.4f}, {hi:+.4f}]  P(diff<=0)={p_gt:.4f}")

    tuned = [n for n in names if n != "dummy_prior"]
    spread_all = (max(res[n]["test"]["roc_auc"] for n in tuned)
                  - min(res[n]["test"]["roc_auc"] for n in tuned))
    mainstream = [n for n in tuned if n != "gaussian_nb"]
    spread_main = (max(res[n]["test"]["roc_auc"] for n in mainstream)
                   - min(res[n]["test"]["roc_auc"] for n in mainstream))
    within = np.mean([res[n]["cv_auc_sd"] for n in mainstream])
    print(f"\nAUC spread across all fitted families      : {spread_all:.4f}")
    print(f"AUC spread excluding naive Bayes           : {spread_main:.4f}")
    print(f"mean within-family 5-fold CV sd            : {within:.4f}")
    print(f"spread / within-family noise (mainstream)  : {spread_main / within:.1f}x")

    json.dump({"per_model": res, "pairwise_auc": diffs,
               "spread_all": spread_all, "spread_mainstream": spread_main,
               "mean_within_family_cv_sd": float(within)},
              open(os.path.join(HERE, "comparison.json"), "w"), indent=2)


def robustness():
    """Re-check the headline gaps under a different resampling scheme/seed:
    repeated 5-fold CV over the FULL dataset with the already-selected configs."""
    from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
    X, y = load()
    cat_mask = [False] * len(NUM) + [True] * len(CAT)
    fixed = {
        "logreg_l2": Pipeline([("prep", prep_dense()),
                               ("clf", LogisticRegression(C=0.3, max_iter=3000))]),
        "decision_tree": Pipeline([("prep", prep_dense()),
                                   ("clf", DecisionTreeClassifier(
                                       max_depth=12, min_samples_leaf=100,
                                       random_state=SEED))]),
        "mlp": Pipeline([("prep", prep_dense()),
                         ("clf", MLPClassifier(hidden_layer_sizes=(128, 64),
                                               alpha=1e-4, max_iter=100,
                                               early_stopping=True,
                                               n_iter_no_change=8,
                                               random_state=SEED))]),
        "random_forest": Pipeline([("prep", prep_dense()),
                                   ("clf", RandomForestClassifier(
                                       n_estimators=400, min_samples_leaf=5,
                                       max_features=0.3, n_jobs=-1,
                                       random_state=SEED))]),
        "hist_gbm": Pipeline([("prep", prep_native()),
                              ("clf", HistGradientBoostingClassifier(
                                  categorical_features=cat_mask, learning_rate=0.05,
                                  max_leaf_nodes=31, min_samples_leaf=20,
                                  max_iter=500, early_stopping=True,
                                  n_iter_no_change=20, random_state=SEED))]),
    }
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=99)
    print("\nrepeated 5x2-fold CV on full data (independent of the 80/20 split):")
    out = {}
    for n, est in fixed.items():
        s = cross_val_score(est, X, y, scoring="roc_auc", cv=cv, n_jobs=-1)
        out[n] = {"mean": float(s.mean()), "sd": float(s.std(ddof=1))}
        print(f"  {n:14s} auc={s.mean():.4f} +/- {s.std(ddof=1):.4f}")
    print(f"  => hist_gbm - logreg_l2 = "
          f"{out['hist_gbm']['mean'] - out['logreg_l2']['mean']:+.4f}")
    json.dump(out, open(os.path.join(HERE, "robustness.json"), "w"), indent=2)
