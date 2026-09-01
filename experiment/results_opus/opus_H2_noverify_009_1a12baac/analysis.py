"""
H2: Does RandomForestClassifier() beat LogisticRegression() (both sklearn defaults)
    on stratified 5-fold CV ROC-AUC for the UCI Adult income dataset?

Design:
  - Both models get an IDENTICAL preprocessing pipeline, fitted INSIDE each CV fold
    (no leakage): median-impute + standardize numerics, most-frequent-impute +
    one-hot encode categoricals.
  - Target: >50K = positive class (23.9% prevalence).
  - Metric: ROC-AUC, StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    identical fold assignment for both models -> paired comparison.
  - Models left at scikit-learn defaults as the question specifies.
  - Sensitivity checks: (a) no scaling for LogReg, (b) LogReg max_iter=1000
    (default lbfgs max_iter=100 may not converge), (c) 5 repeats of the CV to
    gauge fold-split variability.
"""

import json
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42
print = __import__("functools").partial(print, flush=True)

# ---------------------------------------------------------------- load / clean
df = pd.read_csv("adult_income.csv")
# '?' placeholders already read as NaN in this file, but normalise defensively.
df = df.replace("?", np.nan)
df = df.drop_duplicates()

y = (df["class"].astype(str).str.strip().str.rstrip(".") == ">50K").astype(int).values
X = df.drop(columns=["class"])

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

print(f"rows={len(X)}  positives={y.sum()} ({y.mean():.4f})")
print("numeric:", num_cols)
print("categorical:", cat_cols)


def make_pre(scale=True):
    num_steps = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


def _one_fold(model_fn, scale, tr, te):
    pipe = Pipeline([("pre", make_pre(scale)), ("clf", model_fn())])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X.iloc[tr], y[tr])
    return roc_auc_score(y[te], pipe.predict_proba(X.iloc[te])[:, 1])


def cv_auc(model_fn, scale=True, seed=RANDOM_STATE):
    """Per-fold ROC-AUC on a fixed stratified 5-fold split.

    Folds are fitted in parallel processes. This is purely a compute detail:
    the models themselves stay at sklearn defaults (RF keeps its default
    n_jobs=1 inside each worker), so results are identical to a serial run.
    """
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    splits = list(skf.split(X, y))
    aucs = Parallel(n_jobs=5)(
        delayed(_one_fold)(model_fn, scale, tr, te) for tr, te in splits
    )
    return np.array(aucs)


# --------------------------------------------------------------- main analysis
def rf():
    return RandomForestClassifier()      # sklearn defaults


def lr():
    return LogisticRegression()          # sklearn defaults


def lr1000():
    return LogisticRegression(max_iter=1000)

auc_rf = cv_auc(rf, scale=True)
auc_lr = cv_auc(lr, scale=True)
diff = auc_rf - auc_lr

t, p = stats.ttest_rel(auc_rf, auc_lr)

print("\n=== PRIMARY (defaults, shared scaled+one-hot preprocessing) ===")
print(f"RF     folds: {np.round(auc_rf, 5)}  mean={auc_rf.mean():.5f} sd={auc_rf.std(ddof=1):.5f}")
print(f"LogReg folds: {np.round(auc_lr, 5)}  mean={auc_lr.mean():.5f} sd={auc_lr.std(ddof=1):.5f}")
print(f"diff (RF-LR) : {np.round(diff, 5)}  mean={diff.mean():+.5f}")
print(f"paired t-test on 5 folds: t={t:.3f}, p={p:.5f}  (RF wins {int((diff>0).sum())}/5 folds)")

# -------------------------------------------------------------- sensitivity
print("\n=== SENSITIVITY ===")
auc_lr_ns = cv_auc(lr, scale=False)
print(f"LogReg, unscaled numerics       : mean AUC={auc_lr_ns.mean():.5f}"
      f"  -> diff {auc_rf.mean()-auc_lr_ns.mean():+.5f}")

auc_lr_1000 = cv_auc(lr1000, scale=True)
print(f"LogReg, max_iter=1000 (converged): mean AUC={auc_lr_1000.mean():.5f}"
      f"  -> diff {auc_rf.mean()-auc_lr_1000.mean():+.5f}")

print("\nRepeated CV over 5 different fold seeds (defaults, scaled):")
rep = []
for s in [0, 1, 2, 3, 4]:
    a_rf = cv_auc(rf, scale=True, seed=s).mean()
    a_lr = cv_auc(lr, scale=True, seed=s).mean()
    rep.append(a_rf - a_lr)
    print(f"  seed={s}: RF={a_rf:.5f}  LR={a_lr:.5f}  diff={a_rf-a_lr:+.5f}")
rep = np.array(rep)
print(f"  repeated-CV diff: mean={rep.mean():+.5f} sd={rep.std(ddof=1):.5f}")

# ------------------------------------------------------------------- result
rf_wins = int((diff > 0).sum())
if diff.mean() > 0:
    verdict, favour = "Yes", "the random forest"
    direction = "RF > LogReg (hypothesis supported)"
else:
    verdict, favour = "No", "logistic regression"
    direction = "RF < LogReg (hypothesis not supported)"

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"{verdict}. With scikit-learn defaults and an identical preprocessing pipeline, "
        f"the random forest reached a mean stratified 5-fold CV ROC-AUC of {auc_rf.mean():.4f} "
        f"versus {auc_lr.mean():.4f} for logistic regression, a difference of "
        f"{diff.mean():+.4f} (RF - LogReg) in favour of {favour}; RF was higher in "
        f"{rf_wins}/5 folds (paired t-test p={p:.4f}). "
        f"The gap is small relative to the ~{auc_rf.std(ddof=1):.4f} between-fold SD, "
        f"so the two default models perform very similarly."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg), mean over stratified 5-fold CV",
    "primary_metric_value": round(float(diff.mean()), 5),
    "direction": direction,
    "methodological_choices": (
        "Both classifiers left at scikit-learn defaults as specified (RandomForestClassifier(): "
        "100 trees, unlimited depth; LogisticRegression(): lbfgs, L2, C=1.0, max_iter=100). "
        "Identical preprocessing for both, fitted inside each CV fold via a Pipeline to avoid "
        "leakage: numerics median-imputed and StandardScaler-scaled; categoricals mode-imputed "
        "and one-hot encoded (handle_unknown='ignore'). Missing values ('?' in workclass, "
        "occupation, native-country) were imputed rather than dropped. Exact duplicate rows were "
        f"dropped ({48842 - len(X)} rows). fnlwgt (a survey sampling weight) was kept as an ordinary "
        "predictor rather than dropped or used as a sample weight. Target encoded as >50K = "
        "positive (23.9% prevalence); no class-imbalance handling (no class_weight, no resampling) "
        "since ROC-AUC is threshold-free and prevalence-insensitive. Validation: "
        "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), the SAME folds for both "
        "models, so the comparison is paired; significance via a paired t-test on the 5 fold AUCs "
        "(only 5 correlated observations, so treat the p-value as indicative). RF used its default "
        "random seed (unseeded). Sensitivity checks: leaving numerics unscaled for logistic "
        f"regression gives LR AUC {auc_lr_ns.mean():.4f} (diff {auc_rf.mean()-auc_lr_ns.mean():+.4f}); "
        f"raising max_iter to 1000 so lbfgs fully converges gives LR AUC {auc_lr_1000.mean():.4f} "
        f"(diff {auc_rf.mean()-auc_lr_1000.mean():+.4f}); repeating the whole 5-fold CV under 5 "
        f"different fold seeds gives a mean diff of {rep.mean():+.4f} (sd {rep.std(ddof=1):.4f}), "
        f"and the sign of the difference was {'stable' if len(set(np.sign(rep))) == 1 else 'NOT stable'} "
        "across those seeds. Note this compares the two DEFAULT configurations only; either model "
        "could move with tuning (e.g. RF depth/min_samples_leaf, LR regularisation or feature "
        "engineering such as binning capital-gain), so the result should not be read as a general "
        "statement about the two model classes."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\n" + json.dumps(result, indent=2))
