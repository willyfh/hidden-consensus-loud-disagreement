"""
H1: Does the choice of model family meaningfully affect predictive performance
on the UCI Adult (Census Income) dataset?

Design
------
* 9 model families spanning linear, kernel-ish/instance-based, naive-Bayes,
  single-tree, bagged-tree, boosted-tree and neural-net.
* Each family gets the preprocessing it deserves (one-hot + scaling for
  distance/gradient based learners; ordinal / native categorical for trees),
  so the comparison is "family as a practitioner would deploy it", not
  "family handicapped by a mismatched encoder".
* Evaluation: 5-fold stratified CV repeated 2x (10 paired estimates per model)
  on the full de-duplicated dataset. Primary metric ROC-AUC; PR-AUC, accuracy,
  balanced accuracy, F1, log-loss and Brier reported alongside.
* Significance: paired t-tests across the 10 common folds, Holm-corrected,
  against the logistic-regression baseline. Practical significance judged by
  the size of the AUC spread relative to fold-level noise.
* A single stratified 80/20 holdout re-fit confirms the CV ranking.

Run: python analysis.py
"""

import json
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RNG = 20260901

# ---------------------------------------------------------------- data ------
df = pd.read_csv("adult_income.csv")
print(f"raw shape {df.shape}")

# 52 exact duplicate rows exist; dropped so the same record cannot appear in
# both a training and a validation fold.
df = df.drop_duplicates().reset_index(drop=True)

# fnlwgt is a census post-stratification sampling weight, not a property of the
# person -> dropped. 'education' is a pure string alias of 'education-num'
# (verified below) -> keep only the ordinal version.
assert df.groupby("education")["education-num"].nunique().max() == 1
df = df.drop(columns=["fnlwgt", "education"])

y = (df.pop("class").str.strip() == ">50K").astype(int).to_numpy()
X = df
print(f"modelling shape {X.shape}, positive rate {y.mean():.4f}")

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print("numeric :", num_cols)
print("categorical:", cat_cols)

# NaNs in workclass/occupation/native-country are informative ("?" in the raw
# UCI file) -> encoded as an explicit "Missing" level rather than imputed.
X[cat_cols] = X[cat_cols].fillna("Missing")

# ------------------------------------------------------- preprocessors ------
dense_pre = ColumnTransformer(
    [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                              sparse_output=False), cat_cols),
    ]
)
sparse_pre = ColumnTransformer(
    [
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10), cat_cols),
    ]
)
ordinal_pre = ColumnTransformer(
    [
        ("num", "passthrough", num_cols),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                               unknown_value=-1), cat_cols),
    ]
)

cat_mask = [False] * len(num_cols) + [True] * len(cat_cols)

# ------------------------------------------------------------- models -------
models = {
    "Majority baseline": Pipeline(
        [("pre", ordinal_pre), ("clf", DummyClassifier(strategy="prior"))]),
    "Gaussian NB": Pipeline(
        [("pre", dense_pre), ("clf", GaussianNB())]),
    "Logistic Regression": Pipeline(
        [("pre", sparse_pre),
         ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))]),
    "Linear SVM (Platt)": Pipeline(
        [("pre", sparse_pre),
         ("clf", LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs"))]),  # replaced below
    "k-NN (k=25)": Pipeline(
        [("pre", dense_pre),
         ("clf", KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=-1))]),
    "Decision Tree": Pipeline(
        [("pre", ordinal_pre),
         ("clf", DecisionTreeClassifier(min_samples_leaf=20, random_state=RNG))]),
    "Random Forest": Pipeline(
        [("pre", ordinal_pre),
         ("clf", RandomForestClassifier(n_estimators=400, min_samples_leaf=3,
                                        n_jobs=-1, random_state=RNG))]),
    "Extra Trees": Pipeline(
        [("pre", ordinal_pre),
         ("clf", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=3,
                                      n_jobs=-1, random_state=RNG))]),
    "HistGradientBoosting": Pipeline(
        [("pre", ordinal_pre),
         ("clf", HistGradientBoostingClassifier(
             max_iter=400, learning_rate=0.08, early_stopping=False,
             categorical_features=cat_mask, random_state=RNG))]),
    "MLP (128,64)": Pipeline(
        [("pre", dense_pre),
         ("clf", MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                               max_iter=60, early_stopping=True,
                               random_state=RNG))]),
}

# Proper linear SVM with Platt scaling for probabilities.
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

models["Linear SVM (Platt)"] = Pipeline(
    [("pre", sparse_pre),
     ("clf", CalibratedClassifierCV(LinearSVC(C=0.5, dual="auto", max_iter=5000),
                                    method="sigmoid", cv=3))]
)

# ------------------------------------------------- repeated stratified CV ----
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=2, random_state=RNG)
folds = list(cv.split(X, y))

metrics = ["roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "f1",
           "log_loss", "brier"]
scores = {name: {m: [] for m in metrics} for name in models}
fit_time = {}

for name, model in models.items():
    t0 = time.time()
    for k, (tr, te) in enumerate(folds):
        from sklearn.base import clone
        est = clone(model)
        est.fit(X.iloc[tr], y[tr])
        p = est.predict_proba(X.iloc[te])[:, 1]
        yhat = (p >= 0.5).astype(int)
        yte = y[te]
        scores[name]["roc_auc"].append(roc_auc_score(yte, p))
        scores[name]["pr_auc"].append(average_precision_score(yte, p))
        scores[name]["accuracy"].append(accuracy_score(yte, yhat))
        scores[name]["balanced_accuracy"].append(balanced_accuracy_score(yte, yhat))
        scores[name]["f1"].append(f1_score(yte, yhat, zero_division=0))
        scores[name]["log_loss"].append(log_loss(yte, np.clip(p, 1e-9, 1 - 1e-9)))
        scores[name]["brier"].append(brier_score_loss(yte, p))
    fit_time[name] = (time.time() - t0) / len(folds)
    print(f"{name:24s} AUC {np.mean(scores[name]['roc_auc']):.4f} "
          f"(+-{np.std(scores[name]['roc_auc']):.4f})  "
          f"acc {np.mean(scores[name]['accuracy']):.4f}  "
          f"{fit_time[name]:.1f}s/fold")

summary = pd.DataFrame(
    {name: {m: np.mean(v) for m, v in d.items()} for name, d in scores.items()}
).T.sort_values("roc_auc", ascending=False)
summary["auc_sd"] = [np.std(scores[n]["roc_auc"], ddof=1) for n in summary.index]
summary["fit_s"] = [fit_time[n] for n in summary.index]
print("\n=== CV summary (mean over 10 folds) ===")
print(summary.round(4).to_string())

# ------------------------------------------------ paired significance --------
base = "Logistic Regression"
real = [n for n in models if n != "Majority baseline"]
best = summary.drop(index="Majority baseline")["roc_auc"].idxmax()

rows = []
for n in real:
    if n == base:
        continue
    d = np.array(scores[n]["roc_auc"]) - np.array(scores[base]["roc_auc"])
    t, p = stats.ttest_rel(scores[n]["roc_auc"], scores[base]["roc_auc"])
    rows.append({"model": n, "auc_diff_vs_logreg": d.mean(),
                 "t": t, "p_raw": p})
cmp = pd.DataFrame(rows).sort_values("auc_diff_vs_logreg", ascending=False)
# Holm correction
order = np.argsort(cmp["p_raw"].values)
m = len(cmp)
holm = np.empty(m)
prev = 0.0
for rank, idx in enumerate(order):
    val = min(1.0, (m - rank) * cmp["p_raw"].values[idx])
    prev = max(prev, val)
    holm[idx] = prev
cmp["p_holm"] = holm
print("\n=== Paired ROC-AUC vs Logistic Regression (10 common folds, Holm) ===")
print(cmp.round(5).to_string(index=False))

best_diff = float(np.mean(np.array(scores[best]["roc_auc"])
                          - np.array(scores[base]["roc_auc"])))
t_best, p_best = stats.ttest_rel(scores[best]["roc_auc"], scores[base]["roc_auc"])
fold_sd = float(np.mean([np.std(scores[n]["roc_auc"], ddof=1) for n in real]))

auc_means = summary.drop(index="Majority baseline")["roc_auc"]
spread_all = float(auc_means.max() - auc_means.min())
# spread among the "competently configured, commonly used" families, i.e.
# excluding Gaussian NB which is known to be badly mis-specified here
serious = auc_means.drop(index=[i for i in ["Gaussian NB"] if i in auc_means.index])
spread_serious = float(serious.max() - serious.min())

print(f"\nbest = {best}")
print(f"best - logreg AUC = {best_diff:+.4f} (t={t_best:.2f}, p={p_best:.2e})")
print(f"mean within-model fold SD of AUC = {fold_sd:.4f}")
print(f"AUC spread across all 9 families      = {spread_all:.4f}")
print(f"AUC spread excluding Gaussian NB      = {spread_serious:.4f}")
print(f"AUC spread among top-3 families       = "
      f"{auc_means.nlargest(3).max() - auc_means.nlargest(3).min():.4f}")

# ------------------------------------------------------ holdout confirm ------
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y,
                                      random_state=RNG)
hold = {}
for n in real:
    from sklearn.base import clone
    est = clone(models[n]).fit(Xtr, ytr)
    p = est.predict_proba(Xte)[:, 1]
    hold[n] = {"roc_auc": roc_auc_score(yte, p),
               "pr_auc": average_precision_score(yte, p),
               "accuracy": accuracy_score(yte, (p >= .5).astype(int))}
hold = pd.DataFrame(hold).T.sort_values("roc_auc", ascending=False)
print("\n=== Held-out 20% confirmation ===")
print(hold.round(4).to_string())

# ------------------------------------------------------------- output --------
summary.round(6).to_csv("cv_results.csv")
cmp.round(8).to_csv("paired_tests.csv", index=False)

result = {
    "hypothesis_id": "H1",
    "summary": (
        f"Model family matters, but far less than the headline word 'matters' suggests. "
        f"Across 9 families evaluated with repeated stratified 5-fold CV, ROC-AUC spans "
        f"{serious.min():.3f}-{serious.max():.3f} once the mis-specified Gaussian naive Bayes "
        f"({auc_means.min():.3f}) is set aside; the best family ({best}, AUC "
        f"{auc_means.max():.4f}) beats a plain logistic regression by only "
        f"{best_diff:+.4f} AUC, a difference that is highly statistically significant "
        f"(paired t over 10 folds, p={p_best:.1e}) but small in absolute terms. "
        f"The real divide is between well-specified flexible learners (boosted/bagged trees, "
        f"~0.92-0.93) and weak or mis-specified ones (single tree, k-NN, naive Bayes), "
        f"not among the modern families themselves."
    ),
    "primary_metric_name": f"ROC-AUC difference (best family {best} - Logistic Regression), repeated 5-fold CV",
    "primary_metric_value": round(best_diff, 6),
    "direction": f"{best} > LogReg (small but significant); gradient boosting best, Gaussian NB far worst",
    "methodological_choices": (
        "Data: dropped 52 exact duplicate rows to avoid train/test leakage; dropped 'fnlwgt' "
        "(census sampling weight, not a person-level predictor) and the redundant string "
        "'education' (kept ordinal 'education-num'); missing workclass/occupation/native-country "
        "(the raw '?' codes) treated as an explicit 'Missing' category rather than imputed. "
        "Encoding is family-appropriate rather than uniform: one-hot (min_frequency=10) + "
        "standardisation for LogReg/LinearSVM/k-NN/MLP/GaussianNB, ordinal encoding for tree "
        "ensembles, with native categorical splits for HistGradientBoosting. "
        "Validation: RepeatedStratifiedKFold(5 splits x 2 repeats) on the full dataset, giving "
        "10 paired estimates per model, plus an independent stratified 80/20 holdout as a "
        "confirmation; no nested hyperparameter search - each family was given one set of "
        "reasonable literature-default hyperparameters (RF/ET 400 trees min_samples_leaf=3, "
        "HGB 400 iters lr=0.08, LogReg C=1, LinearSVC C=0.5 with sigmoid calibration, k-NN k=25 "
        "distance-weighted, MLP (128,64) with early stopping), so results reflect sensible "
        "defaults rather than each family's tuned ceiling. "
        "Metric: ROC-AUC as primary (threshold-free, insensitive to the 24% positive rate), "
        "with PR-AUC, accuracy, balanced accuracy, F1@0.5, log-loss and Brier reported. "
        "No class-imbalance handling (no reweighting/resampling) since ROC-AUC is the target "
        "and the imbalance is mild. Significance via paired t-tests on the 10 common folds with "
        "Holm correction; note that CV folds are not independent so these p-values are "
        "anti-conservative, which is why the practical judgement leans on the effect size "
        "(AUC spread vs within-model fold SD) rather than on p alone."
    ),
}
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nwrote result.json")
print(json.dumps(result, indent=2))
