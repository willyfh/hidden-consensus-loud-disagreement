"""
H6: Is the model well-calibrated?  --  UCI Adult (Census Income)

Approach
--------
Fit three classifiers of different families on a stratified 75/25 train/test
split, then assess the calibration of their predicted P(income > 50K) on the
held-out test set with:

  * Reliability curves (equal-frequency bins)
  * Expected / Maximum Calibration Error (ECE / MCE, 15 quantile bins)
  * Brier score + Murphy decomposition (reliability / resolution / uncertainty)
  * Spiegelhalter's z-test (a proper calibration hypothesis test)
  * A parametric-bootstrap null for ECE, because ECE is biased away from 0 in
    finite samples even for a perfectly calibrated model.  This gives a p-value
    that accounts for that bias.
  * Calibration slope/intercept from a logistic recalibration on the logit
    of the predicted probability (slope 1, intercept 0 == well calibrated).
  * How much post-hoc calibration (Platt / isotonic, fitted on a separate
    calibration split) actually improves log-loss and Brier.

The "primary" model is the gradient-boosted tree (HistGradientBoosting), the
best-performing model here and the one a practitioner would most likely deploy;
logistic regression and random forest are reported alongside it.
"""

import json
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RNG = 42
rng = np.random.default_rng(RNG)
EPS = 1e-12

# ----------------------------------------------------------------- data ------
df = pd.read_csv("adult_income.csv")
# a handful of dupes exist; drop them so train/test cannot share identical rows
df = df.drop_duplicates().reset_index(drop=True)

y = (df["class"].str.strip() == ">50K").astype(int).to_numpy()
X = df.drop(columns=["class", "fnlwgt"])  # fnlwgt is a survey sampling weight, not a feature
num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
print(f"n={len(df)}  positives={y.mean():.4f}  num={num_cols}  cat={cat_cols}")

# train / calibration / test = 60 / 15 / 25, stratified
X_tr_full, X_te, y_tr_full, y_te = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RNG
)
X_tr, X_cal, y_tr, y_cal = train_test_split(
    X_tr_full, y_tr_full, test_size=0.20, stratify=y_tr_full, random_state=RNG
)
print(f"train={len(y_tr)} cal={len(y_cal)} test={len(y_te)}")

# ------------------------------------------------------------ preprocess -----
def make_prep(scale, sparse=True):
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
                        (
                            "oh",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=10,
                                sparse_output=sparse,
                            ),
                        ),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


models = {
    "logreg": Pipeline(
        [("prep", make_prep(True)), ("clf", LogisticRegression(max_iter=2000, C=1.0))]
    ),
    "hgb": Pipeline(
        [
            ("prep", make_prep(False, sparse=False)),
            ("clf", HistGradientBoostingClassifier(random_state=RNG)),
        ]
    ),
    "rf": Pipeline(
        [
            ("prep", make_prep(False, sparse=False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=400, min_samples_leaf=1, n_jobs=-1, random_state=RNG
                ),
            ),
        ]
    ),
}

# ------------------------------------------------ calibration diagnostics ----
def quantile_bins(p, n_bins):
    """Equal-frequency bin edges; robust to ties (Adult probs have mass points)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    return np.digitize(p, edges[1:-1], right=True)


def reliability(p, y, n_bins=15):
    b = quantile_bins(p, n_bins)
    rows = []
    for k in np.unique(b):
        m = b == k
        rows.append(
            dict(
                bin=int(k),
                n=int(m.sum()),
                mean_pred=float(p[m].mean()),
                frac_pos=float(y[m].mean()),
                gap=float(y[m].mean() - p[m].mean()),
            )
        )
    return pd.DataFrame(rows)


def ece_mce(p, y, n_bins=15):
    r = reliability(p, y, n_bins)
    w = r["n"] / r["n"].sum()
    return float((w * r["gap"].abs()).sum()), float(r["gap"].abs().max())


def spiegelhalter_z(p, y):
    """z = sum((y-p)(1-2p)) / sqrt(sum((1-2p)^2 p(1-p))); ~N(0,1) if calibrated."""
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    from scipy.stats import norm

    z = num / den
    return float(z), float(2 * norm.sf(abs(z)))


def ece_pvalue(p, y, n_bins=15, n_sim=2000):
    """Parametric-bootstrap null: resample labels y* ~ Bernoulli(p) and recompute
    ECE.  Corrects for the positive finite-sample bias of ECE."""
    obs, _ = ece_mce(p, y, n_bins)
    sims = np.empty(n_sim)
    for i in range(n_sim):
        ys = (rng.random(len(p)) < p).astype(int)
        sims[i], _ = ece_mce(p, ys, n_bins)
    return obs, float(sims.mean()), float(np.quantile(sims, 0.95)), float(
        (sims >= obs).mean()
    )


def cal_slope_intercept(p, y):
    """Logistic recalibration on logit(p): slope=1 & intercept=0 iff calibrated.
    slope < 1 => over-confident (probabilities too extreme)."""
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    lo = np.log(pc / (1 - pc)).reshape(-1, 1)
    lr = LogisticRegression(max_iter=1000, C=1e12).fit(lo, y)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def brier_decomposition(p, y, n_bins=15):
    """Murphy: BS = reliability - resolution + uncertainty."""
    r = reliability(p, y, n_bins)
    w = (r["n"] / r["n"].sum()).to_numpy()
    ybar = y.mean()
    rel = float(np.sum(w * (r["mean_pred"] - r["frac_pos"]) ** 2))
    res = float(np.sum(w * (r["frac_pos"] - ybar) ** 2))
    unc = float(ybar * (1 - ybar))
    return rel, res, unc


# --------------------------------------------------------------- run ---------
results = {}
report_lines = []
for name, pipe in models.items():
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]

    ece, mce = ece_mce(p, y_te)
    obs_ece, null_ece, null_q95, p_ece = ece_pvalue(p, y_te)
    z, p_z = spiegelhalter_z(np.clip(p, EPS, 1 - EPS), y_te)
    slope, icpt = cal_slope_intercept(p, y_te)
    rel, res, unc = brier_decomposition(p, y_te)

    # post-hoc recalibration fitted on the held-out calibration split
    post = {}
    for method in ("sigmoid", "isotonic"):
        cc = CalibratedClassifierCV(pipe, method=method, cv="prefit").fit(X_cal, y_cal)
        pp = cc.predict_proba(X_te)[:, 1]
        e2, _ = ece_mce(pp, y_te)
        post[method] = dict(
            ece=e2,
            brier=float(brier_score_loss(y_te, pp)),
            logloss=float(log_loss(y_te, np.clip(pp, EPS, 1 - EPS))),
            auc=float(roc_auc_score(y_te, pp)),
        )

    results[name] = dict(
        auc=float(roc_auc_score(y_te, p)),
        brier=float(brier_score_loss(y_te, p)),
        logloss=float(log_loss(y_te, np.clip(p, EPS, 1 - EPS))),
        ece=ece,
        mce=mce,
        ece_null_mean=null_ece,
        ece_null_q95=null_q95,
        ece_pvalue=p_ece,
        spiegelhalter_z=z,
        spiegelhalter_p=p_z,
        cal_slope=slope,
        cal_intercept=icpt,
        brier_reliability=rel,
        brier_resolution=res,
        brier_uncertainty=unc,
        mean_pred=float(p.mean()),
        base_rate=float(y_te.mean()),
        post_hoc=post,
    )

    line = (
        f"\n=== {name} ===\n"
        f"AUC {results[name]['auc']:.4f} | Brier {results[name]['brier']:.4f} | "
        f"LogLoss {results[name]['logloss']:.4f}\n"
        f"ECE {ece:.4f} (null mean {null_ece:.4f}, null 95% {null_q95:.4f}, p={p_ece:.4f}) | MCE {mce:.4f}\n"
        f"Spiegelhalter z {z:+.3f} (p={p_z:.4g}) | slope {slope:.3f} intercept {icpt:+.3f}\n"
        f"Brier decomp: reliability {rel:.5f}  resolution {res:.5f}  uncertainty {unc:.5f}\n"
        f"mean pred {p.mean():.4f} vs base rate {y_te.mean():.4f}\n"
        f"post-hoc  Platt ECE {post['sigmoid']['ece']:.4f} / LL {post['sigmoid']['logloss']:.4f}"
        f" | isotonic ECE {post['isotonic']['ece']:.4f} / LL {post['isotonic']['logloss']:.4f}"
    )
    print(line)
    print(reliability(p, y_te).to_string(index=False))
    report_lines.append(line)

# -------------------------------------------- robustness across 6 splits ----
# A single split could be lucky, so refit the primary model on 6 different
# stratified splits and re-measure ECE / Spiegelhalter z / calibration slope.
seed_rows = []
for s in [0, 1, 2, 3, 4, 42]:
    Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.25, stratify=y, random_state=s)
    pipe = Pipeline(
        [
            ("prep", make_prep(False, sparse=False)),
            ("clf", HistGradientBoostingClassifier(random_state=s)),
        ]
    ).fit(Xa, ya)
    ps = pipe.predict_proba(Xb)[:, 1]
    e, _ = ece_mce(ps, yb)
    nullmean = float(
        np.mean([ece_mce(ps, (rng.random(len(ps)) < ps).astype(int))[0] for _ in range(400)])
    )
    zz, pzz = spiegelhalter_z(np.clip(ps, EPS, 1 - EPS), yb)
    sl, ic = cal_slope_intercept(ps, yb)
    seed_rows.append(
        dict(seed=s, auc=float(roc_auc_score(yb, ps)), brier=float(brier_score_loss(yb, ps)),
             ece=e, ece_null=nullmean, z=zz, p_z=pzz, slope=sl, intercept=ic)
    )
seed_df = pd.DataFrame(seed_rows)
print("\n=== robustness across splits (hgb) ===")
print(seed_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
print(
    f"mean ECE {seed_df.ece.mean():.4f} | mean null ECE {seed_df.ece_null.mean():.4f} "
    f"| mean slope {seed_df.slope.mean():.3f}"
)

# ---------------------------------------------------------- conclusion -------
prim = "hgb"
r = results[prim]
print("\n" + "=" * 70)
print(
    f"PRIMARY MODEL = {prim}: ECE={r['ece']:.4f} vs perfectly-calibrated null "
    f"{r['ece_null_mean']:.4f} (p={r['ece_pvalue']:.4f})"
)

summary = (
    f"Yes for the deployed gradient-boosted model, no for a random forest -- calibration is a "
    f"property of the model, not the dataset. HistGradientBoosting's test ECE of {r['ece']:.4f} "
    f"is statistically indistinguishable from the {r['ece_null_mean']:.4f} a perfectly calibrated "
    f"model of the same size would show by chance (p={r['ece_pvalue']:.2f}; Spiegelhalter z="
    f"{r['spiegelhalter_z']:+.2f}, p={r['spiegelhalter_p']:.2f}; calibration slope "
    f"{r['cal_slope']:.2f}), and post-hoc recalibration does not improve its log-loss "
    f"({r['logloss']:.4f} -> {r['post_hoc']['sigmoid']['logloss']:.4f}). Logistic regression is "
    f"nearly as good (ECE={results['logreg']['ece']:.4f}), whereas the random forest is badly "
    f"over-confident (ECE={results['rf']['ece']:.4f}, slope={results['rf']['cal_slope']:.2f}, "
    f"z={results['rf']['spiegelhalter_z']:+.1f}) and isotonic recalibration cuts its log-loss from "
    f"{results['rf']['logloss']:.4f} to {results['rf']['post_hoc']['isotonic']['logloss']:.4f}."
)

out = {
    "hypothesis_id": "H6",
    "summary": summary,
    "primary_metric_name": (
        "Expected Calibration Error (15 equal-frequency bins) of the primary "
        "HistGradientBoosting model on the held-out test set"
    ),
    "primary_metric_value": round(r["ece"], 5),
    "direction": (
        "model is well-calibrated (gradient boosting: ECE indistinguishable from the "
        "perfectly-calibrated null, p=%.2f); the random forest comparator is miscalibrated "
        "and over-confident" % r["ece_pvalue"]
    ),
    "methodological_choices": (
        "Rows deduplicated (48,790 left) and 'fnlwgt' dropped as a survey sampling weight rather "
        "than a predictor; target binarised to >50K (23.9% positive). Stratified 60/15/25 "
        "train/calibration/test split (seed 42), the 15% calibration split held out solely for "
        "fitting the post-hoc Platt/isotonic recalibrators so that they are never evaluated on "
        "their own fitting data. Median/most-frequent imputation of the NaNs in workclass, "
        "occupation and native-country (treating missingness as ignorable rather than as its own "
        "'Missing' level), one-hot encoding with min_frequency=10 to fold rare native-country "
        "levels together. Primary model: HistGradientBoostingClassifier at sklearn defaults, with "
        "LogisticRegression (C=1) and RandomForest (400 trees, unrestricted leaves) as "
        "comparators; no hyperparameter tuning and no class-imbalance handling (no class_weight, "
        "no resampling) because reweighting would deliberately destroy calibration. Calibration "
        "judged by ECE with 15 EQUAL-FREQUENCY bins -- equal-width bins would give a very "
        "different, smaller-looking number here since most predictions sit near 0 -- plus MCE, "
        "Spiegelhalter's z-test, the Murphy reliability term of the Brier decomposition, the "
        "logistic calibration slope/intercept, and the change in log-loss under post-hoc "
        "recalibration. Crucially, significance for ECE comes from a 2000-draw parametric "
        "bootstrap null (labels resampled as Bernoulli(p_hat)) rather than from comparing ECE to "
        "0, because ECE is biased upward in finite samples; without that null the model's "
        "ECE=0.007 could be misread as evidence of miscalibration. Conclusion re-checked across "
        "6 random splits. Another researcher might reasonably pick a different primary model "
        "(the answer flips for random forest), equal-width bins, a different bin count, "
        "cross-validated rather than single-split probabilities, or a Hosmer-Lemeshow test."
    ),
    "_detail": results,
    "_robustness_across_splits": seed_rows,
}
with open("result.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nwrote result.json\n" + summary)
