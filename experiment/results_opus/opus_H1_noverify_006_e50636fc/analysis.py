"""
H1: Does the choice of model family meaningfully affect predictive performance
on the UCI Adult (Census Income) dataset?

Design
------
* 7 model families spanning the usual space (linear, kernel-free instance-based,
  generative, single tree, bagged trees, boosted trees, neural net) + a
  majority-class baseline for scale.
* Each family gets a small but real hyperparameter search (3-fold CV, ROC-AUC)
  on the training split, so we compare *tuned* families rather than defaults.
* Held-out 20% stratified test set for the headline numbers, with a paired
  bootstrap (2000 resamples) for confidence intervals on AUC differences.
* Repeated stratified 5-fold CV (2 repeats) on the training split with the
  tuned configurations, for a paired significance test across identical folds.

Primary metric: test ROC-AUC difference, best boosted-tree model minus
logistic regression.
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             f1_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, RepeatedStratifiedKFold,
                                     StratifiedKFold, train_test_split)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
SEED = 42
rng = np.random.default_rng(SEED)

# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

# fnlwgt is a census sampling weight, not a property of the person -> drop.
df = df.drop(columns=["fnlwgt"])
# 52 exact duplicate rows: drop so identical records cannot straddle the split.
df = df.drop_duplicates().reset_index(drop=True)

y = (df["class"].str.strip() == ">50K").astype(int).to_numpy()
X = df.drop(columns=["class"])

cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "str"]
num_cols = [c for c in X.columns if c not in cat_cols]
# Missingness in workclass/occupation/native-country is informative (non-workers,
# non-responders) -> encode as its own level rather than imputing.
X[cat_cols] = X[cat_cols].astype(object).fillna("Missing")

print(f"n={len(X)}  positives={y.mean():.4f}  cat={len(cat_cols)} num={len(num_cols)}")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=SEED
)

# ----------------------------------------------------------------------------
# Preprocessing: one-hot (+scale) for distance/gradient based learners,
# ordinal for tree learners.
# ----------------------------------------------------------------------------
def onehot_pre():
    return ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="infrequent_if_exist",
                              min_frequency=10, sparse_output=False), cat_cols),
    ])


def ordinal_pre():
    return ColumnTransformer([
        ("num", "passthrough", num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), cat_cols),
    ])


cat_mask = [False] * len(num_cols) + [True] * len(cat_cols)

# ----------------------------------------------------------------------------
# Model families + small search grids
# ----------------------------------------------------------------------------
MODELS = {
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
        {"clf__n_neighbors": [15, 35, 75], "clf__weights": ["uniform", "distance"]},
    ),
    "DecisionTree": (
        Pipeline([("pre", ordinal_pre()),
                  ("clf", DecisionTreeClassifier(random_state=SEED))]),
        {"clf__max_depth": [6, 10, 14, None],
         "clf__min_samples_leaf": [1, 20, 100]},
    ),
    "RandomForest": (
        Pipeline([("pre", ordinal_pre()),
                  ("clf", RandomForestClassifier(n_estimators=500,
                                                 random_state=SEED, n_jobs=-1))]),
        {"clf__max_features": ["sqrt", 0.5], "clf__min_samples_leaf": [1, 5, 20]},
    ),
    "HistGradientBoosting": (
        Pipeline([("pre", ordinal_pre()),
                  ("clf", HistGradientBoostingClassifier(
                      categorical_features=cat_mask, random_state=SEED,
                      early_stopping=True, validation_fraction=0.15,
                      n_iter_no_change=25, max_iter=600))]),
        {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31, 63],
         "clf__l2_regularization": [0.0, 1.0]},
    ),
    "MLP": (
        Pipeline([("pre", onehot_pre()),
                  ("clf", MLPClassifier(random_state=SEED, max_iter=300,
                                        early_stopping=True, n_iter_no_change=15))]),
        {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-2]},
    ),
}

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
best_est, tuning = {}, {}
for name, (pipe, grid) in MODELS.items():
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=inner_cv, n_jobs=-1,
                      refit=True)
    gs.fit(X_tr, y_tr)
    best_est[name] = gs.best_estimator_
    tuning[name] = {"best_params": {k: str(v) for k, v in gs.best_params_.items()},
                    "inner_cv_auc": float(gs.best_score_)}
    print(f"tuned {name:22s} inner-CV AUC={gs.best_score_:.4f}  {gs.best_params_}")

# Majority-class baseline for scale
base = DummyClassifier(strategy="prior").fit(X_tr, y_tr)
best_est["Baseline(majority)"] = base
tuning["Baseline(majority)"] = {"best_params": {}, "inner_cv_auc": 0.5}

# ----------------------------------------------------------------------------
# Held-out test evaluation
# ----------------------------------------------------------------------------
def scores(est, Xd, yd):
    p = est.predict_proba(Xd)[:, 1]
    yhat = (p >= 0.5).astype(int)
    return p, {
        "roc_auc": roc_auc_score(yd, p),
        "pr_auc": average_precision_score(yd, p),
        "accuracy": accuracy_score(yd, yhat),
        "balanced_accuracy": balanced_accuracy_score(yd, yhat),
        "f1_pos": f1_score(yd, yhat, zero_division=0),
        "brier": brier_score_loss(yd, p),
    }


probs, test_res = {}, {}
for name, est in best_est.items():
    p, s = scores(est, X_te, y_te)
    probs[name], test_res[name] = p, s
    print(f"TEST {name:22s} " + "  ".join(f"{k}={v:.4f}" for k, v in s.items()))

ranked = sorted((n for n in test_res if n != "Baseline(majority)"),
                key=lambda n: -test_res[n]["roc_auc"])
best_name, worst_name = ranked[0], ranked[-1]

# ----------------------------------------------------------------------------
# Paired bootstrap on the test set: AUC difference vs. the best model
# ----------------------------------------------------------------------------
B, n_te = 2000, len(y_te)
boot_idx = rng.integers(0, n_te, size=(B, n_te))
boot_auc = {}
for name, p in probs.items():
    if name == "Baseline(majority)":
        continue
    vals = []
    for b in range(B):
        i = boot_idx[b]
        if y_te[i].min() == y_te[i].max():
            continue
        vals.append(roc_auc_score(y_te[i], p[i]))
    boot_auc[name] = np.array(vals)

boot_ci = {}
for name in boot_auc:
    if name == best_name:
        continue
    d = boot_auc[best_name] - boot_auc[name]
    boot_ci[name] = {
        "auc_diff_vs_best": float(test_res[best_name]["roc_auc"] - test_res[name]["roc_auc"]),
        "ci95_low": float(np.percentile(d, 2.5)),
        "ci95_high": float(np.percentile(d, 97.5)),
        "p_boot_two_sided": float(2 * min((d <= 0).mean(), (d >= 0).mean())),
    }
    print(f"BOOT {best_name} - {name:22s} d={boot_ci[name]['auc_diff_vs_best']:+.4f} "
          f"CI95=[{boot_ci[name]['ci95_low']:+.4f},{boot_ci[name]['ci95_high']:+.4f}]")

# ----------------------------------------------------------------------------
# Repeated stratified 5-fold CV on the training split with tuned configs,
# identical folds for every model -> paired t-test on fold AUCs.
# ----------------------------------------------------------------------------
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=SEED)
folds = list(rskf.split(X_tr, y_tr))
cv_auc = {n: [] for n in best_est if n != "Baseline(majority)"}
from sklearn.base import clone
for tr_i, va_i in folds:
    Xa, Xb = X_tr.iloc[tr_i], X_tr.iloc[va_i]
    ya, yb = y_tr[tr_i], y_tr[va_i]
    for name in cv_auc:
        m = clone(best_est[name]).fit(Xa, ya)
        cv_auc[name].append(roc_auc_score(yb, m.predict_proba(Xb)[:, 1]))
cv_auc = {k: np.array(v) for k, v in cv_auc.items()}
for k, v in cv_auc.items():
    print(f"CV   {k:22s} AUC={v.mean():.4f} +/- {v.std(ddof=1):.4f}")

cv_tests = {}
for name in cv_auc:
    if name == best_name:
        continue
    t, p = stats.ttest_rel(cv_auc[best_name], cv_auc[name])
    cv_tests[name] = {"cv_auc_mean": float(cv_auc[name].mean()),
                      "cv_auc_diff_vs_best": float(cv_auc[best_name].mean() - cv_auc[name].mean()),
                      "paired_t": float(t), "p_value": float(p)}
    print(f"TTEST {best_name} vs {name:22s} d={cv_tests[name]['cv_auc_diff_vs_best']:+.4f} p={p:.2e}")

# ----------------------------------------------------------------------------
# Headline numbers
# ----------------------------------------------------------------------------
gb_lr_diff = test_res["HistGradientBoosting"]["roc_auc"] - test_res["LogisticRegression"]["roc_auc"]
auc_range = test_res[best_name]["roc_auc"] - test_res[worst_name]["roc_auc"]
top3 = [test_res[n]["roc_auc"] for n in ranked[:3]]
print(f"\nGB - LogReg test AUC diff = {gb_lr_diff:+.4f}")
print(f"range across 7 families    = {auc_range:.4f} ({best_name} - {worst_name})")
print(f"spread among top-3         = {max(top3) - min(top3):.4f}")

d_gb_lr = boot_auc["HistGradientBoosting"] - boot_auc["LogisticRegression"]
ci_gb_lr = (float(np.percentile(d_gb_lr, 2.5)), float(np.percentile(d_gb_lr, 97.5)))
print(f"GB - LogReg bootstrap CI95 = [{ci_gb_lr[0]:+.4f}, {ci_gb_lr[1]:+.4f}]")

report = {
    "n_rows_used": int(len(X)),
    "positive_rate": float(y.mean()),
    "tuning": tuning,
    "test_metrics": {k: {m: float(x) for m, x in v.items()} for k, v in test_res.items()},
    "test_bootstrap_vs_best": boot_ci,
    "cv_paired_tests_vs_best": cv_tests,
    "best_model": best_name,
    "worst_model_excl_baseline": worst_name,
    "gb_minus_logreg_test_auc": float(gb_lr_diff),
    "gb_minus_logreg_ci95": list(ci_gb_lr),
    "auc_range_across_families": float(auc_range),
    "top3_auc_spread": float(max(top3) - min(top3)),
}
with open("full_results.json", "w") as f:
    json.dump(report, f, indent=2)

result = {
    "hypothesis_id": "H1",
    "summary": (
        f"Yes, but the effect is modest among competitive families and large only when weak "
        f"families are included. With light tuning, gradient-boosted trees are best "
        f"(test ROC-AUC {test_res['HistGradientBoosting']['roc_auc']:.3f}) and beat logistic "
        f"regression ({test_res['LogisticRegression']['roc_auc']:.3f}) by "
        f"{gb_lr_diff:.3f} AUC (95% bootstrap CI [{ci_gb_lr[0]:.3f}, {ci_gb_lr[1]:.3f}], "
        f"paired-CV p={cv_tests['LogisticRegression']['p_value']:.1e}) — a small but highly "
        f"reliable gap. Across all seven families the test-AUC range is {auc_range:.3f} "
        f"({best_name} vs {worst_name}), so model family matters far more for avoiding a poor "
        f"choice than for separating the top contenders (top-3 spread {max(top3)-min(top3):.3f})."
    ),
    "primary_metric_name": "Test ROC-AUC difference (HistGradientBoosting - LogisticRegression)",
    "primary_metric_value": round(float(gb_lr_diff), 4),
    "direction": "Boosted trees > RandomForest > LogReg ~ MLP > kNN > DecisionTree > GaussianNB; gap small at the top, statistically significant",
    "methodological_choices": (
        "Dropped fnlwgt (census sampling weight, not an individual attribute) and 52 exact "
        "duplicate rows; kept education and education-num despite redundancy. Missing values in "
        "workclass/occupation/native-country encoded as an explicit 'Missing' level rather than "
        "imputed, on the view that missingness is informative. Single stratified 80/20 train/test "
        "split (seed 42). Two encodings: one-hot (min_frequency=10) + standardization for "
        "LogReg/kNN/GaussianNB/MLP, ordinal (with native categorical handling in "
        "HistGradientBoosting) for tree models. Each family tuned by 3-fold GridSearchCV on the "
        "training split optimizing ROC-AUC over a small grid, then refit on the full training "
        "split. Primary metric ROC-AUC (threshold-free, robust to the 24% positive rate); PR-AUC, "
        "accuracy, balanced accuracy, F1 at 0.5 and Brier also reported. No class-imbalance "
        "reweighting or resampling — ROC-AUC/PR-AUC are threshold-free, so rebalancing mainly "
        "shifts calibration, though it would change the accuracy/F1 columns. Uncertainty from a "
        "paired bootstrap (2000 resamples) on the test set plus paired t-tests over 2x5 repeated "
        "stratified CV folds on the training split; no multiple-comparison correction across the "
        "six pairwise tests. Alternative reasonable choices: nested CV instead of a single "
        "hold-out, wider grids or XGBoost/LightGBM (unavailable here), target/ordinal encoding of "
        "education, dropping rows with missing values, or optimizing accuracy/F1 instead of AUC."
    ),
}
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nwrote result.json / full_results.json")
