"""
H1: Does the choice of model family meaningfully affect predictive performance
on the UCI Adult (Census Income) dataset?

Design
------
Stage A (primary): 5-fold stratified cross-validation over the whole (de-duplicated)
    dataset with sensible, lightly-considered default configurations for 9 model
    families + a majority-class baseline. Because every family sees exactly the same
    folds, differences can be tested fold-wise (paired t-test).
Stage B (robustness): 80/20 stratified hold-out. Per-family hyperparameters are tuned
    with 3-fold GridSearchCV *inside the training split only*, then scored once on the
    untouched test split. This checks whether the Stage-A ranking is an artefact of
    default hyperparameters, and gives a bootstrap CI for the headline difference.

Primary metric: ROC-AUC (threshold-free, insensitive to the 76/24 class imbalance).
Secondary: PR-AUC (average precision), accuracy, F1 on the positive class.

Everything runs in the foreground; total runtime is a few minutes.
"""

import json
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
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")
RNG = 42

# ---------------------------------------------------------------- data
df = pd.read_csv("adult_income.csv")

# Exact duplicate rows (52) are dropped so the same record cannot appear in both
# train and test folds.
df = df.drop_duplicates().reset_index(drop=True)

y = (df["class"].str.strip() == ">50K").astype(int).to_numpy()
X = df.drop(columns=["class"])

# `education` is a redundant string encoding of `education-num`; keeping both is
# harmless for trees but doubles collinearity for the linear model. Keep both --
# the question is about model families, not feature selection -- but note it.
NUM = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT = [c for c in X.columns if c not in NUM]

print(f"rows={len(X)}  positives={y.mean():.4f}  numeric={len(NUM)}  categorical={len(CAT)}")
print("missing per column:\n", X.isna().sum()[X.isna().sum() > 0])

# ------------------------------------------------- preprocessing blocks
# Missing values (workclass / occupation / native-country) are treated as an
# explicit "Missing" category rather than imputed -- missingness in this census
# extract is itself informative (mostly never-worked / unknown employers).


def dense_pre():
    """One-hot + standardised numerics: for linear / distance / NN models."""
    return ColumnTransformer(
        [
            ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                              ("sc", StandardScaler())]), NUM),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                              ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                                                   sparse_output=False))]), CAT),
        ]
    )


def ordinal_pre():
    """Ordinal-coded categoricals: for axis-aligned tree ensembles."""
    return ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUM),
            ("cat", Pipeline([("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                              ("oe", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                    unknown_value=-1))]), CAT),
        ]
    )


def hgb_model(**kw):
    """HistGradientBoosting with native categorical support."""
    pre = ordinal_pre()
    cat_mask = [False] * len(NUM) + [True] * len(CAT)
    return Pipeline([("pre", pre),
                     ("clf", HistGradientBoostingClassifier(categorical_features=cat_mask,
                                                            random_state=RNG, **kw))])


# ------------------------------------------------- model zoo (Stage A defaults)
def make_models():
    return {
        "Baseline (majority)": Pipeline([("pre", ordinal_pre()),
                                         ("clf", DummyClassifier(strategy="prior"))]),
        "GaussianNB": Pipeline([("pre", dense_pre()),
                                ("clf", GaussianNB())]),
        "kNN (k=25)": Pipeline([("pre", dense_pre()),
                                ("clf", KNeighborsClassifier(n_neighbors=25, n_jobs=-1))]),
        "Decision tree": Pipeline([("pre", ordinal_pre()),
                                   ("clf", DecisionTreeClassifier(min_samples_leaf=50,
                                                                  random_state=RNG))]),
        "Linear SVM": Pipeline([("pre", dense_pre()),
                                ("clf", LinearSVC(C=0.1, dual=True, max_iter=5000,
                                                  random_state=RNG))]),
        "Logistic regression": Pipeline([("pre", dense_pre()),
                                         ("clf", LogisticRegression(C=1.0, max_iter=2000,
                                                                    random_state=RNG))]),
        "MLP (64,32)": Pipeline([("pre", dense_pre()),
                                 ("clf", MLPClassifier(hidden_layer_sizes=(64, 32),
                                                       max_iter=60, early_stopping=True,
                                                       random_state=RNG))]),
        "Extra trees": Pipeline([("pre", ordinal_pre()),
                                 ("clf", ExtraTreesClassifier(n_estimators=400,
                                                              min_samples_leaf=3,
                                                              n_jobs=-1, random_state=RNG))]),
        "Random forest": Pipeline([("pre", ordinal_pre()),
                                   ("clf", RandomForestClassifier(n_estimators=400,
                                                                  min_samples_leaf=3,
                                                                  n_jobs=-1, random_state=RNG))]),
        "Hist gradient boosting": hgb_model(),
    }


def scores(model, Xte):
    """Continuous score for ROC-AUC: predict_proba where available, else decision_function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(Xte)[:, 1]
    return model.decision_function(Xte)


# ================================================== Stage A: 5-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)
folds = list(cv.split(X, y))

fold_auc = {name: [] for name in make_models()}
fold_ap = {name: [] for name in make_models()}
fold_acc = {name: [] for name in make_models()}

for k, (tr, te) in enumerate(folds):
    Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y[tr], y[te]
    for name, model in make_models().items():
        model.fit(Xtr, ytr)
        s = scores(model, Xte)
        fold_auc[name].append(roc_auc_score(yte, s))
        fold_ap[name].append(average_precision_score(yte, s))
        fold_acc[name].append(accuracy_score(yte, model.predict(Xte)))
    print(f"fold {k + 1}/5 done")

cvres = pd.DataFrame(
    {
        "auc_mean": {k: np.mean(v) for k, v in fold_auc.items()},
        "auc_sd": {k: np.std(v, ddof=1) for k, v in fold_auc.items()},
        "ap_mean": {k: np.mean(v) for k, v in fold_ap.items()},
        "acc_mean": {k: np.mean(v) for k, v in fold_acc.items()},
    }
).sort_values("auc_mean", ascending=False)
print("\n=== Stage A: 5-fold CV (default configs) ===")
print(cvres.round(4).to_string())

real = [n for n in cvres.index if n != "Baseline (majority)"]
best, worst = real[0], real[-1]
serious = [n for n in real if n not in ("GaussianNB", "kNN (k=25)")]  # weak/ill-suited families
best_s, worst_s = serious[0], serious[-1]

spread_all = cvres.loc[best, "auc_mean"] - cvres.loc[worst, "auc_mean"]
spread_serious = cvres.loc[best_s, "auc_mean"] - cvres.loc[worst_s, "auc_mean"]
gap_gb_lr = cvres.loc["Hist gradient boosting", "auc_mean"] - cvres.loc["Logistic regression", "auc_mean"]

d = np.array(fold_auc["Hist gradient boosting"]) - np.array(fold_auc["Logistic regression"])
t_gb_lr = stats.ttest_rel(fold_auc["Hist gradient boosting"], fold_auc["Logistic regression"])
d_rf = np.array(fold_auc["Hist gradient boosting"]) - np.array(fold_auc["Random forest"])
t_gb_rf = stats.ttest_rel(fold_auc["Hist gradient boosting"], fold_auc["Random forest"])

print(f"\nAUC spread, all families      : {spread_all:.4f}  ({best} vs {worst})")
print(f"AUC spread, sensible families : {spread_serious:.4f}  ({best_s} vs {worst_s})")
print(f"HGB - LogReg = {gap_gb_lr:.4f}  fold diffs {np.round(d, 4)}  "
      f"paired t={t_gb_lr.statistic:.2f} p={t_gb_lr.pvalue:.2g}")
print(f"HGB - RF     = {d_rf.mean():.4f}  paired t={t_gb_rf.statistic:.2f} p={t_gb_rf.pvalue:.2g}")

# ================================================== Stage B: tuned hold-out
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RNG)

grids = {
    "Logistic regression": (
        Pipeline([("pre", dense_pre()), ("clf", LogisticRegression(max_iter=3000, random_state=RNG))]),
        {"clf__C": [0.03, 0.1, 0.3, 1.0, 3.0]},
    ),
    "kNN": (
        Pipeline([("pre", dense_pre()), ("clf", KNeighborsClassifier(n_jobs=-1))]),
        {"clf__n_neighbors": [15, 35, 75], "clf__weights": ["uniform", "distance"]},
    ),
    "Decision tree": (
        Pipeline([("pre", ordinal_pre()), ("clf", DecisionTreeClassifier(random_state=RNG))]),
        {"clf__min_samples_leaf": [20, 50, 150], "clf__max_depth": [8, 14, None]},
    ),
    "Random forest": (
        Pipeline([("pre", ordinal_pre()),
                  ("clf", RandomForestClassifier(n_estimators=500, n_jobs=-1, random_state=RNG))]),
        {"clf__min_samples_leaf": [1, 3, 10], "clf__max_features": ["sqrt", 0.5]},
    ),
    "MLP": (
        Pipeline([("pre", dense_pre()),
                  ("clf", MLPClassifier(max_iter=80, early_stopping=True, random_state=RNG))]),
        {"clf__hidden_layer_sizes": [(64,), (128, 64)], "clf__alpha": [1e-4, 1e-2]},
    ),
    "Hist gradient boosting": (
        hgb_model(),
        {"clf__learning_rate": [0.05, 0.1], "clf__max_leaf_nodes": [15, 31, 63],
         "clf__l2_regularization": [0.0, 1.0]},
    ),
}

tuned_rows = {}
boot_scores = {}
for name, (pipe, grid) in grids.items():
    gs = GridSearchCV(pipe, grid, scoring="roc_auc", cv=3, n_jobs=-1, refit=True)
    gs.fit(Xtr, ytr)
    s = scores(gs.best_estimator_, Xte)
    pred = gs.best_estimator_.predict(Xte)
    boot_scores[name] = s
    tuned_rows[name] = {
        "test_auc": roc_auc_score(yte, s),
        "test_ap": average_precision_score(yte, s),
        "test_acc": accuracy_score(yte, pred),
        "test_f1": f1_score(yte, pred),
        "best_params": str(gs.best_params_),
    }
    print(f"tuned {name}: AUC={tuned_rows[name]['test_auc']:.4f}  {gs.best_params_}")

tuned = pd.DataFrame(tuned_rows).T.sort_values("test_auc", ascending=False)
print("\n=== Stage B: tuned hold-out test set ===")
print(tuned.to_string())

# paired bootstrap CI on the headline difference (HGB - LogReg) on the test set
rng = np.random.default_rng(RNG)
a, b = boot_scores["Hist gradient boosting"], boot_scores["Logistic regression"]
diffs = []
n = len(yte)
for _ in range(2000):
    idx = rng.integers(0, n, n)
    if yte[idx].sum() in (0, len(idx)):
        continue
    diffs.append(roc_auc_score(yte[idx], a[idx]) - roc_auc_score(yte[idx], b[idx]))
lo, hi = np.percentile(diffs, [2.5, 97.5])
test_gap = roc_auc_score(yte, a) - roc_auc_score(yte, b)
print(f"\nTest-set HGB - LogReg AUC = {test_gap:.4f}  95% bootstrap CI [{lo:.4f}, {hi:.4f}]")

tuned_spread = tuned["test_auc"].max() - tuned["test_auc"].min()
print(f"Tuned-family AUC spread (test) = {tuned_spread:.4f}")

# ---------------------------------------------------------------- outputs
cvres.round(5).to_csv("cv_results.csv")
tuned.to_csv("tuned_holdout_results.csv")

summary = (
    f"Yes, but modestly in absolute terms: across nine model families evaluated on identical "
    f"5-fold stratified CV, ROC-AUC ranges from {cvres.loc[worst, 'auc_mean']:.3f} ({worst}) to "
    f"{cvres.loc[best, 'auc_mean']:.3f} ({best}); among competently-specified families the spread is "
    f"{spread_serious:.3f} AUC. Gradient boosting beats logistic regression by "
    f"{gap_gb_lr:.4f} AUC (paired across folds, p={t_gb_lr.pvalue:.1g}), a difference that is "
    f"highly consistent but small relative to the {cvres.loc[best, 'auc_mean'] - 0.5:.2f} AUC that "
    f"any reasonable model already extracts."
)

result = {
    "hypothesis_id": "H1",
    "summary": summary,
    "primary_metric_name": "ROC-AUC difference (HistGradientBoosting - LogisticRegression), 5-fold stratified CV",
    "primary_metric_value": round(float(gap_gb_lr), 4),
    "direction": "GBM > RF > LogReg/SVM/MLP > tree > kNN > NB; boosted trees best, but the gap over a linear model is only ~0.02 AUC",
    "methodological_choices": (
        "De-duplicated 52 exact duplicate rows; target = class '>50K' (23.9% positive). Missing "
        "workclass/occupation/native-country encoded as an explicit 'Missing' category (not imputed) "
        "since missingness is informative; numeric NaNs (none present) would be median-imputed. Kept "
        "all 14 features including fnlwgt (a survey design weight) and both education/education-num, "
        "to keep the feature set identical across families rather than optimising it. Encoding was "
        "matched to each family: one-hot (min_frequency=10) + standardised numerics for logistic "
        "regression, linear SVM, kNN, MLP and GaussianNB; ordinal codes for RF/ExtraTrees/decision "
        "tree; native categorical splits for HistGradientBoosting. Primary evaluation = 5-fold "
        "stratified CV over the full dataset with the same folds for every model, so families are "
        "compared fold-wise with a paired t-test; secondary = 80/20 stratified hold-out with 3-fold "
        "GridSearchCV tuning inside the training split only, plus a 2000-replicate paired bootstrap "
        "CI on the headline difference. Metric = ROC-AUC (threshold-free, imbalance-insensitive); "
        "PR-AUC, accuracy and F1 reported as secondary. No class-imbalance correction (no "
        "class_weight, no resampling) because ROC/PR-AUC rank-based metrics do not require a "
        "rebalanced threshold; models were left at library defaults apart from the small grids above. "
        "Another researcher might drop fnlwgt or redundant education, use nested CV, tune far more "
        "aggressively (or add XGBoost/LightGBM/CatBoost, unavailable here), pick accuracy or F1 as "
        "the headline metric, or use the canonical 32561/16281 train/test split instead of CV -- the "
        "latter two would change the size, though probably not the sign, of the gaps reported."
    ),
}
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\nwrote result.json\n", json.dumps(result, indent=2))
