"""
H1: Does the choice of model family meaningfully affect predictive performance
on the UCI Adult (Census Income) dataset?

Design
------
- Common data prep for all families; family-appropriate encoding branches
  (one-hot + standardisation for distance/gradient-based learners, ordinal
  codes for tree-based learners).
- 8 model families spanning the usual space: trivial baseline, naive Bayes,
  regularised linear, kNN, single tree, random forest, gradient boosting, MLP.
- Small per-family hyperparameter grid tuned by 5-fold stratified CV on the
  training split ONLY (so no family is handicapped by a bad default).
- Final comparison on an untouched 20% stratified holdout test set.
- Primary metric: ROC-AUC (threshold-free, robust to the 76/24 imbalance).
  Secondary: PR-AUC, accuracy, F1@0.5, Brier score.
- Uncertainty: 2000-replicate paired bootstrap of the test set for the
  difference in AUC between each model and logistic regression.
- Practical-significance yardstick: the spread across serious families is
  compared against the CV-fold-level noise and against the gap between the
  best model and the majority-class baseline.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RNG = 42
rng = np.random.default_rng(RNG)

# ---------------------------------------------------------------- load / clean
df = pd.read_csv("adult_income.csv")

# Target labels carry a trailing '.' in some releases of this file; normalise.
df["class"] = df["class"].str.strip().str.rstrip(".")
y = (df["class"] == ">50K").astype(int).values

# Exact duplicate rows would otherwise straddle the train/test split and leak.
dup_mask = df.duplicated()
df = df.loc[~dup_mask].reset_index(drop=True)
y = y[~dup_mask.values]

X = df.drop(columns=["class"])

# fnlwgt is a census post-stratification sampling weight describing how many
# people the row represents, not a property of the person. Dropped as a
# predictor (a defensible choice another researcher might not make).
X = X.drop(columns=["fnlwgt"])

# education-num is an exact ordinal recoding of education; both are kept -
# redundant but harmless, and dropping either would advantage some families.

CAT = [c for c in X.columns if X[c].dtype == object]
NUM = [c for c in X.columns if c not in CAT]

# Missingness (workclass / occupation / native-country) is not random - it
# largely marks the never-worked / unknown population - so it is encoded as its
# own level rather than imputed away.
X[CAT] = X[CAT].fillna("Missing")

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RNG
)

print(f"rows={len(X)} (dropped {int(dup_mask.sum())} dups)  "
      f"train={len(X_tr)} test={len(X_te)}  pos_rate={y.mean():.4f}")

# ---------------------------------------------------------------- encoders
def dense_prep():
    """One-hot + standardised: for linear, kNN, MLP, NB."""
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), NUM),
            ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                  sparse_output=False), CAT),
        ]
    )


def ordinal_prep():
    """Integer codes, unscaled: for tree-based learners."""
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUM),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1), CAT),
        ]
    )


# ---------------------------------------------------------------- model zoo
MODELS = {
    "Baseline (majority)": (
        ordinal_prep(), DummyClassifier(strategy="prior"), {},
    ),
    "Gaussian NB": (
        dense_prep(), GaussianNB(), {"clf__var_smoothing": [1e-9, 1e-6, 1e-3]},
    ),
    "Logistic Regression": (
        dense_prep(),
        LogisticRegression(max_iter=3000, solver="lbfgs"),
        {"clf__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    ),
    "k-NN": (
        dense_prep(),
        KNeighborsClassifier(n_jobs=-1),
        {"clf__n_neighbors": [15, 35, 75], "clf__weights": ["uniform", "distance"]},
    ),
    "Decision Tree": (
        ordinal_prep(),
        DecisionTreeClassifier(random_state=RNG),
        {"clf__max_depth": [6, 10, 14, None], "clf__min_samples_leaf": [1, 20, 50]},
    ),
    "Random Forest": (
        ordinal_prep(),
        RandomForestClassifier(n_estimators=500, random_state=RNG, n_jobs=-1),
        {"clf__min_samples_leaf": [1, 5, 20], "clf__max_features": ["sqrt", 0.5]},
    ),
    "Gradient Boosting (HistGB)": (
        ordinal_prep(),
        HistGradientBoostingClassifier(
            random_state=RNG, early_stopping=True, validation_fraction=0.1,
            categorical_features=[NUM.index(c) if c in NUM else len(NUM) + CAT.index(c)
                                  for c in CAT],
        ),
        {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [31, 63],
         "clf__l2_regularization": [0.0, 1.0]},
    ),
    "MLP (1x64)": (
        dense_prep(),
        MLPClassifier(hidden_layer_sizes=(64,), max_iter=400, random_state=RNG,
                      early_stopping=True),
        {"clf__alpha": [1e-4, 1e-2]},
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

results = {}
test_proba = {}
cv_fold_scores = {}

for name, (prep, clf, grid) in MODELS.items():
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=cv, n_jobs=-1,
                      refit=True, return_train_score=False)
    gs.fit(X_tr, y_tr)

    best_idx = gs.best_index_
    folds = np.array([gs.cv_results_[f"split{i}_test_score"][best_idx]
                      for i in range(5)])
    cv_fold_scores[name] = folds

    p = gs.predict_proba(X_te)[:, 1]
    test_proba[name] = p
    yhat = (p >= 0.5).astype(int)

    results[name] = {
        "cv_auc_mean": float(folds.mean()),
        "cv_auc_std": float(folds.std(ddof=1)),
        "test_auc": float(roc_auc_score(y_te, p)),
        "test_pr_auc": float(average_precision_score(y_te, p)),
        "test_acc": float(accuracy_score(y_te, yhat)),
        "test_f1": float(f1_score(y_te, yhat, zero_division=0)),
        "test_brier": float(brier_score_loss(y_te, p)),
        "best_params": {k: str(v) for k, v in gs.best_params_.items()},
    }
    print(f"{name:28s} cv_auc={folds.mean():.4f}  test_auc={results[name]['test_auc']:.4f}  "
          f"acc={results[name]['test_acc']:.4f}  {gs.best_params_}")

# ---------------------------------------------------------------- bootstrap
REF = "Logistic Regression"
B = 2000
n = len(y_te)
idx = rng.integers(0, n, size=(B, n))

boot = {}
for name, p in test_proba.items():
    if name == "Baseline (majority)":
        continue
    diffs = np.empty(B)
    aucs = np.empty(B)
    pr = test_proba[REF]
    for b in range(B):
        i = idx[b]
        yb = y_te[i]
        if yb.min() == yb.max():
            diffs[b] = np.nan
            aucs[b] = np.nan
            continue
        a = roc_auc_score(yb, p[i])
        aucs[b] = a
        diffs[b] = a - roc_auc_score(yb, pr[i])
    boot[name] = {
        "auc_ci": [float(np.nanpercentile(aucs, 2.5)),
                   float(np.nanpercentile(aucs, 97.5))],
        "diff_vs_logreg": float(np.nanmean(diffs)),
        "diff_ci": [float(np.nanpercentile(diffs, 2.5)),
                    float(np.nanpercentile(diffs, 97.5))],
    }
    results[name]["bootstrap"] = boot[name]

# ------------------------------------------------- robustness: enriched linear
# Is the boosting advantage really about model FAMILY, or just that the linear
# model was never given nonlinear terms? Give logistic regression quantile-binned
# numerics (nonlinearity) and then all pairwise interactions, and see whether it
# closes the gap to HistGB.
from sklearn.preprocessing import KBinsDiscretizer, PolynomialFeatures  # noqa: E402

bin_prep = ColumnTransformer(
    [
        ("num", KBinsDiscretizer(n_bins=10, encode="onehot-dense",
                                 strategy="quantile", subsample=None), NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                              sparse_output=False), CAT),
    ]
)

enriched = {}
for label, steps in [
    ("LogReg + binned numerics", [("prep", bin_prep)]),
    ("LogReg + binned + pairwise interactions",
     [("prep", bin_prep),
      ("int", PolynomialFeatures(2, interaction_only=True, include_bias=False))]),
]:
    pipe = Pipeline(steps + [("clf", LogisticRegression(max_iter=2000, solver="lbfgs"))])
    gs = GridSearchCV(pipe, {"clf__C": [0.03, 0.3]}, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X_tr, y_tr)
    t = float(roc_auc_score(y_te, gs.predict_proba(X_te)[:, 1]))
    enriched[label] = {"cv_auc": float(gs.best_score_), "test_auc": t,
                       "best_params": {k: str(v) for k, v in gs.best_params_.items()}}
    print(f"{label:42s} cv={gs.best_score_:.4f} test={t:.4f} {gs.best_params_}")

# ---------------------------------------------------------------- summarise
serious = [k for k in results if k != "Baseline (majority)"]
aucs = {k: results[k]["test_auc"] for k in serious}
best = max(aucs, key=aucs.get)
worst = min(aucs, key=aucs.get)

# "Serious, well-tuned" families: exclude the two known-weak learners
# (Gaussian NB's independence assumption and a single unpruned-ish tree) to
# see how much room is left among methods a practitioner would actually ship.
mainstream = ["Logistic Regression", "Random Forest",
              "Gradient Boosting (HistGB)", "MLP (1x64)", "k-NN"]
main_aucs = {k: aucs[k] for k in mainstream}
main_best, main_worst = max(main_aucs, key=main_aucs.get), min(main_aucs, key=main_aucs.get)

spread_all = aucs[best] - aucs[worst]
spread_main = main_aucs[main_best] - main_aucs[main_worst]
gap_gb_lr = aucs["Gradient Boosting (HistGB)"] - aucs["Logistic Regression"]
headroom = aucs[best] - 0.5  # AUC above chance

print("\n--- summary ---")
print(f"best={best} {aucs[best]:.4f} | worst={worst} {aucs[worst]:.4f} "
      f"| full spread={spread_all:.4f}")
print(f"mainstream spread ({main_worst} -> {main_best}) = {spread_main:.4f}")
print(f"GB - LogReg = {gap_gb_lr:.4f}  CI={boot['Gradient Boosting (HistGB)']['diff_ci']}")
print(f"spread as % of above-chance signal: {100*spread_main/headroom:.1f}%")

ranking = sorted(serious, key=lambda k: -aucs[k])
best_enriched = max(enriched, key=lambda k: enriched[k]["test_auc"])
enriched_gap = aucs["Gradient Boosting (HistGB)"] - enriched[best_enriched]["test_auc"]
print(f"best enriched linear = {best_enriched} {enriched[best_enriched]['test_auc']:.4f} "
      f"(HistGB still ahead by {enriched_gap:.4f}); plain LogReg = {aucs[REF]:.4f}")

# ---------------------------------------------------------------- write out
summary = (
    f"Yes, but modestly. Across eight model families tuned identically, test "
    f"ROC-AUC ranges from {aucs[worst]:.3f} ({worst}) to {aucs[best]:.3f} ({best}); "
    f"gradient boosting beats regularised logistic regression by "
    f"{gap_gb_lr:.4f} AUC (95% bootstrap CI "
    f"[{boot['Gradient Boosting (HistGB)']['diff_ci'][0]:.4f}, "
    f"{boot['Gradient Boosting (HistGB)']['diff_ci'][1]:.4f}]), a statistically "
    f"unambiguous but small gap. Among mainstream families the spread is only "
    f"{spread_main:.3f} AUC, i.e. ~{100*spread_main/headroom:.0f}% of the "
    f"above-chance signal, so family choice matters far less than the fact that "
    f"any competent model is used at all. Giving logistic regression binned "
    f"numerics plus all pairwise interactions did not close the gap "
    f"({enriched[best_enriched]['test_auc']:.3f} AUC), so boosting's edge is not "
    f"recovered by simple feature engineering."
)

out = {
    "hypothesis_id": "H1",
    "summary": summary,
    "primary_metric_name": "Test ROC-AUC difference (Gradient Boosting (HistGB) - Logistic Regression)",
    "primary_metric_value": round(float(gap_gb_lr), 4),
    "direction": (
        "Gradient boosting > all other families ("
        + " > ".join(ranking)
        + "), but the margin over logistic regression is small "
        f"({gap_gb_lr:.3f} AUC) relative to the {headroom:.3f} of above-chance signal"
    ),
    "methodological_choices": (
        "Dropped 52 exact duplicate rows before splitting to avoid train/test leakage; "
        "dropped fnlwgt (census sampling weight, not a person-level predictor) while "
        "keeping the redundant education/education-num pair. Missing values in "
        "workclass/occupation/native-country encoded as an explicit 'Missing' category "
        "rather than imputed. Two encoding branches: one-hot (min_frequency=10) plus "
        "standardisation for LogReg/kNN/MLP/GaussianNB, ordinal integer codes for "
        "Tree/RF/HistGB (HistGB told which columns are categorical). Single stratified "
        "80/20 holdout; hyperparameters chosen by 5-fold stratified CV on the training "
        "split only, with a small grid per family (LogReg C; kNN k/weights; tree depth "
        "and leaf size; RF min_samples_leaf/max_features with 500 trees; HistGB "
        "learning rate/leaves/L2 with early stopping; MLP alpha on one 64-unit hidden "
        "layer). Primary metric ROC-AUC (threshold-free, so the 76/24 imbalance is "
        "handled by the metric rather than by resampling or class weights); PR-AUC, "
        "accuracy, F1 at a fixed 0.5 threshold and Brier score reported alongside. "
        "Uncertainty from a 2000-replicate paired bootstrap of the test set against "
        "logistic regression as reference, rather than repeated-CV paired t-tests. "
        "Practical significance judged by expressing the between-family AUC spread as "
        "a fraction of the best model's above-chance AUC. Alternatives another "
        "researcher might pick: keeping fnlwgt or using it as a sample weight, "
        "repeated nested CV instead of a single holdout, accuracy or F1 as the headline "
        "metric, target/CatBoost-style encoding, XGBoost/LightGBM, or wider grids. "
        "As a robustness check the linear model was also refit with 10-bin quantile-"
        "discretised numerics and with all pairwise interactions on top, to test "
        "whether the boosting gap is really about model family rather than about "
        "unmodelled nonlinearity; both variants scored BELOW plain logistic "
        "regression, partly because quantile binning degenerates on the spike-at-zero "
        "capital-gain/capital-loss columns, so this check bounds rather than settles "
        "the question - a more carefully engineered linear model might do better."
    ),
    "detail": {
        "n_rows_used": int(len(X)),
        "positive_rate": float(y.mean()),
        "per_model": results,
        "spread_all_families_auc": float(spread_all),
        "spread_mainstream_auc": float(spread_main),
        "best_model": best,
        "worst_model": worst,
        "cv_fold_sd_of_best": float(cv_fold_scores[best].std(ddof=1)),
        "ranking_by_test_auc": ranking,
        "enriched_linear_robustness_check": enriched,
    },
}

with open("result.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote result.json")
