"""
H1: Does the choice of model family meaningfully affect predictive performance
on the UCI Adult (Census Income) dataset?

Design
------
* Target: class (>50K = positive, 23.9% prevalence).
* Primary metric: ROC-AUC (threshold-free, insensitive to the 76/24 imbalance).
  Secondary: PR-AUC (average precision), accuracy, balanced accuracy, F1@0.5.
* 7 model families spanning the usual space: majority-class baseline, Gaussian
  naive Bayes, k-NN, single decision tree, L2 logistic regression, random
  forest, histogram gradient boosting, and a small MLP.
* Stratified 70/30 split. Light hyperparameter tuning by 3-fold CV *inside*
  the training set only; final numbers on the untouched 30% test set with
  paired bootstrap CIs on the between-family differences.
* Verification: 3x5 repeated stratified CV (3 different seeds) over the whole
  dataset, independent of the single split above.

Preprocessing choices
---------------------
* fnlwgt dropped: it is a census sampling weight, not a person-level feature.
* `education` (string) dropped in favour of `education-num` (its exact ordinal
  encoding) to avoid a duplicated signal.
* '?' is already parsed as NaN by pandas; categorical NaN -> explicit
  "Missing" level (missingness is informative here), numeric NaN: none exist.
* Linear / distance / neural models: one-hot (dense) + standard scaling.
  Tree models: ordinal codes (no scaling). Each family gets the encoding it
  is normally given, so the comparison is between families as they'd actually
  be used, not between families forced through one encoder.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RNG = 20260902

# ---------------------------------------------------------------- data ------
df = pd.read_csv("adult_income.csv")
df = df.drop(columns=["fnlwgt", "education"])
y = (df.pop("class").str.strip() == ">50K").astype(int).values
X = df

CAT = [c for c in X.columns if X[c].dtype == object]
NUM = [c for c in X.columns if c not in CAT]
X[CAT] = X[CAT].fillna("Missing")
print(f"n={len(X)}  pos rate={y.mean():.4f}  numeric={NUM}  cat={CAT}")


def onehot_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                              sparse_output=False), CAT),
    ])


def ordinal_pre():
    return ColumnTransformer([
        ("num", "passthrough", NUM),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), CAT),
    ])


# Each family: (pipeline, small tuning grid). Grids are deliberately small and
# comparably sized so no family gets an unfair search budget.
def families(seed):
    return {
        "Baseline (majority)": (
            Pipeline([("p", ordinal_pre()),
                      ("m", DummyClassifier(strategy="prior"))]), {}),
        "GaussianNB": (
            Pipeline([("p", onehot_pre()), ("m", GaussianNB())]),
            {"m__var_smoothing": [1e-9, 1e-6, 1e-3]}),
        "k-NN": (
            Pipeline([("p", onehot_pre()),
                      ("m", KNeighborsClassifier(n_jobs=-1))]),
            {"m__n_neighbors": [15, 35, 75], "m__weights": ["distance"]}),
        "Decision tree": (
            Pipeline([("p", ordinal_pre()),
                      ("m", DecisionTreeClassifier(random_state=seed))]),
            {"m__max_depth": [6, 10, 16, None],
             "m__min_samples_leaf": [1, 20]}),
        "Logistic regression": (
            Pipeline([("p", onehot_pre()),
                      ("m", LogisticRegression(max_iter=3000,
                                               random_state=seed))]),
            {"m__C": [0.03, 0.3, 1.0, 3.0]}),
        "Random forest": (
            Pipeline([("p", ordinal_pre()),
                      ("m", RandomForestClassifier(n_estimators=500,
                                                   n_jobs=-1,
                                                   random_state=seed))]),
            {"m__min_samples_leaf": [1, 5, 20],
             "m__max_features": ["sqrt", 0.5]}),
        "Hist gradient boosting": (
            Pipeline([("p", ordinal_pre()),
                      ("m", HistGradientBoostingClassifier(
                          categorical_features=[len(NUM) + i
                                                for i in range(len(CAT))],
                          random_state=seed))]),
            {"m__learning_rate": [0.05, 0.1],
             "m__max_leaf_nodes": [15, 31, 63],
             "m__max_iter": [300]}),
        "MLP (64,32)": (
            Pipeline([("p", onehot_pre()),
                      ("m", MLPClassifier(hidden_layer_sizes=(64, 32),
                                          max_iter=60, early_stopping=True,
                                          random_state=seed))]),
            {"m__alpha": [1e-4, 1e-2]}),
    }


# ------------------------------------------------- stage 1: single split ----
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.30, stratify=y,
                                      random_state=RNG)
inner = StratifiedKFold(5, shuffle=True, random_state=RNG)

fitted, test_scores, test_prob = {}, {}, {}
for name, (pipe, grid) in families(RNG).items():
    if grid:
        gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=3, n_jobs=-1)
        gs.fit(Xtr, ytr)
        best, params = gs.best_estimator_, gs.best_params_
    else:
        best, params = pipe.fit(Xtr, ytr), {}
    cv_auc = cross_val_score(best, Xtr, ytr, cv=inner, scoring="roc_auc",
                             n_jobs=-1)
    p = best.predict_proba(Xte)[:, 1]
    pred = (p >= 0.5).astype(int)
    test_prob[name] = p
    fitted[name] = best
    test_scores[name] = dict(
        cv_auc_mean=cv_auc.mean(), cv_auc_sd=cv_auc.std(),
        test_auc=roc_auc_score(yte, p),
        test_ap=average_precision_score(yte, p),
        test_acc=accuracy_score(yte, pred),
        test_bacc=balanced_accuracy_score(yte, pred),
        test_f1=f1_score(yte, pred),
        params=str(params))
    print(f"{name:24s} cvAUC={cv_auc.mean():.4f}  testAUC="
          f"{test_scores[name]['test_auc']:.4f}  {params}")

res = pd.DataFrame(test_scores).T.sort_values("test_auc", ascending=False)
print("\n=== held-out test (30 pct, n=%d) ===\n" % len(yte),
      res[["cv_auc_mean", "test_auc", "test_ap", "test_acc", "test_bacc",
           "test_f1"]].to_string())

real = [n for n in res.index if n != "Baseline (majority)"]
best_name, worst_name = res.loc[real, "test_auc"].idxmax(), res.loc[real, "test_auc"].idxmin()
lr_name = "Logistic regression"
spread = res.loc[best_name, "test_auc"] - res.loc[worst_name, "test_auc"]
gap_lr = res.loc[best_name, "test_auc"] - res.loc[lr_name, "test_auc"]
print(f"\nspread best-worst ({best_name} - {worst_name}) = {spread:.4f}")
print(f"gap {best_name} - {lr_name} = {gap_lr:.4f}")

# ----------------------------- paired bootstrap CI on the test differences --
rs = np.random.default_rng(RNG)
idx = np.arange(len(yte))
boot = {k: [] for k in ["best_vs_worst", "best_vs_lr", "lr_vs_worst"]}
pairs = {"best_vs_worst": (best_name, worst_name),
         "best_vs_lr": (best_name, lr_name),
         "lr_vs_worst": (lr_name, worst_name)}
B = 2000
for _ in range(B):
    b = rs.choice(idx, len(idx), replace=True)
    if yte[b].std() == 0:
        continue
    for k, (a, c) in pairs.items():
        boot[k].append(roc_auc_score(yte[b], test_prob[a][b])
                       - roc_auc_score(yte[b], test_prob[c][b]))
print("\n=== paired bootstrap (2000 resamples) AUC differences ===")
boot_ci = {}
for k, (a, c) in pairs.items():
    v = np.array(boot[k])
    boot_ci[k] = (float(v.mean()), float(np.percentile(v, 2.5)),
                  float(np.percentile(v, 97.5)))
    print(f"{a} - {c}: {v.mean():+.4f}  95% CI [{boot_ci[k][1]:+.4f},"
          f" {boot_ci[k][2]:+.4f}]")

# ------------------------------ stage 2: verification, repeated CV, 3 seeds -
print("\n=== verification: 3x5 repeated stratified CV on the full dataset ===")
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=777)
verify, fold_auc = {}, {}
for name, (pipe, _) in families(777).items():
    if name == "Baseline (majority)":
        continue
    # freeze the hyperparameters chosen on the training split above
    est = fitted[name]
    s = cross_val_score(est, X, y, cv=rcv, scoring="roc_auc", n_jobs=-1)
    fold_auc[name] = s
    verify[name] = dict(mean=float(s.mean()), sd=float(s.std()),
                        lo=float(s.min()), hi=float(s.max()))
    print(f"{name:24s} AUC={s.mean():.4f} +/- {s.std():.4f} "
          f"[{s.min():.4f},{s.max():.4f}]")

v = pd.DataFrame(verify).T.sort_values("mean", ascending=False)
v_best, v_worst = v.index[0], v.index[-1]
v_spread = v.loc[v_best, "mean"] - v.loc[v_worst, "mean"]
v_gap_lr = v.loc[v_best, "mean"] - v.loc[lr_name, "mean"]
print(f"\nCV spread {v_best} - {v_worst} = {v_spread:.4f}")
print(f"CV gap {v_best} - {lr_name} = {v_gap_lr:.4f}")
print(f"CV within-model sd (max across families) = {v['sd'].max():.4f}")

# fold-level paired comparisons: identical folds, so differences are paired
print("\n=== paired per-fold differences over the 15 CV folds ===")
paired = {}
for a, c in [(v_best, lr_name), (v_best, "Random forest"),
             (v_best, v_worst), (lr_name, v_worst)]:
    d = fold_auc[a] - fold_auc[c]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    paired[f"{a} - {c}"] = dict(mean=float(d.mean()), sd=float(d.std(ddof=1)),
                                wins=int((d > 0).sum()), n=int(len(d)),
                                t=float(t))
    print(f"{a} - {c}: {d.mean():+.4f} (sd {d.std(ddof=1):.4f}), "
          f"wins {int((d > 0).sum())}/{len(d)} folds, paired t={t:.1f}")


out = {
    "test": res.to_dict(orient="index"),
    "bootstrap": boot_ci,
    "verify_cv": verify, "paired_folds": paired,
    "spread_test": float(spread), "gap_lr_test": float(gap_lr),
    "spread_cv": float(v_spread), "gap_lr_cv": float(v_gap_lr),
}
with open("analysis_details.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("\nwrote analysis_details.json")
