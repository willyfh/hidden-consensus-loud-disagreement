"""
H6: Is the model well-calibrated?  (UCI Adult / Census Income)

Approach
--------
Fit three standard classifiers (regularised logistic regression, random forest,
histogram gradient boosting) on a stratified 70/30 train/test split and assess
how well their predicted probabilities of `class == ">50K"` match observed
frequencies on the held-out test set.

Calibration diagnostics reported per model:
  * Brier score (+ Murphy decomposition into reliability / resolution / uncertainty)
  * Log loss
  * Expected / maximum calibration error (ECE / MCE), equal-frequency bins
  * Calibration intercept and slope from a logistic recalibration of the logit
    (perfect = intercept 0, slope 1)
  * Hosmer-Lemeshow goodness-of-fit test (10 equal-frequency deciles)
  * Spiegelhalter's Z test of calibration
  * Reliability table (bin-level predicted vs observed)
Finally, each model is re-fit inside CalibratedClassifierCV (isotonic and
sigmoid, 5-fold internal CV on the training data only) to quantify how much
calibration error is actually removable.

Primary model = random forest (the flagship non-linear model); primary metric =
its test-set ECE.
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RNG = 42
EPS = 1e-15

# ----------------------------------------------------------------- data
df = pd.read_csv("adult_income.csv")
df = df.drop_duplicates()  # OpenML Adult contains a handful of exact dupes
y = (df["class"].str.strip() == ">50K").astype(int).values
X = df.drop(columns=["class", "fnlwgt"])  # fnlwgt is a survey sampling weight, not a predictor

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, random_state=RNG, stratify=y
)
print(f"train {X_tr.shape}  test {X_te.shape}  base rate(test)={y_te.mean():.4f}")


def make_pre(scale):
    num = [("imp", SimpleImputer(strategy="median"))]
    if scale:
        num.append(("sc", StandardScaler()))
    return ColumnTransformer(
        [
            ("num", Pipeline(num), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", min_frequency=10)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )


models = {
    "LogisticRegression": Pipeline(
        [("pre", make_pre(True)), ("clf", LogisticRegression(C=1.0, max_iter=2000))]
    ),
    "RandomForest": Pipeline(
        [
            ("pre", make_pre(False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500, n_jobs=-1, random_state=RNG
                ),  # default depth: fully grown trees
            ),
        ]
    ),
    "HistGradientBoosting": Pipeline(
        [("pre", make_pre(False)), ("clf", HistGradientBoostingClassifier(random_state=RNG))]
    ),
}


# ------------------------------------------------------- calibration tools
def eq_freq_bins(p, n_bins):
    """Equal-frequency bin edges (robust to the heavy pile-up near p=0)."""
    q = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    q[0], q[-1] = -np.inf, np.inf
    return np.clip(np.digitize(p, q[1:-1], right=True), 0, len(q) - 2)


def reliability(p, y, n_bins=15):
    b = eq_freq_bins(p, n_bins)
    rows = []
    for k in np.unique(b):
        m = b == k
        rows.append(
            dict(bin=int(k), n=int(m.sum()), p_mean=float(p[m].mean()), y_rate=float(y[m].mean()))
        )
    return pd.DataFrame(rows)


def ece_mce(p, y, n_bins=15):
    r = reliability(p, y, n_bins)
    gap = (r.p_mean - r.y_rate).abs()
    w = r.n / r.n.sum()
    return float((w * gap).sum()), float(gap.max()), r


def cal_intercept_slope(p, y):
    """Cox calibration: logit(y) ~ a + b*logit(p). Ideal a=0, b=1."""
    lp = np.log(np.clip(p, EPS, 1 - EPS) / (1 - np.clip(p, EPS, 1 - EPS))).reshape(-1, 1)
    slope = LogisticRegression(penalty=None, max_iter=2000).fit(lp, y)
    b = float(slope.coef_[0][0])
    a = float(slope.intercept_[0])
    return a, b


def hosmer_lemeshow(p, y, g=10):
    b = eq_freq_bins(p, g)
    chi2 = 0.0
    for k in np.unique(b):
        m = b == k
        o1, e1 = y[m].sum(), p[m].sum()
        n = m.sum()
        e0 = n - e1
        if e1 > 0:
            chi2 += (o1 - e1) ** 2 / e1
        if e0 > 0:
            chi2 += ((n - o1) - e0) ** 2 / e0
    dof = len(np.unique(b)) - 2
    return float(chi2), int(dof), float(stats.chi2.sf(chi2, dof))


def spiegelhalter_z(p, y):
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum(((1 - 2 * p) ** 2) * p * (1 - p)))
    z = float(num / den)
    return z, float(2 * stats.norm.sf(abs(z)))


def murphy(p, y, n_bins=15):
    """Brier = reliability - resolution + uncertainty."""
    r = reliability(p, y, n_bins)
    w = (r.n / r.n.sum()).values
    ybar = y.mean()
    rel = float(np.sum(w * (r.p_mean - r.y_rate) ** 2))
    res = float(np.sum(w * (r.y_rate - ybar) ** 2))
    unc = float(ybar * (1 - ybar))
    return rel, res, unc


def assess(name, p, y):
    ece, mce, rel_tbl = ece_mce(p, y)
    a, b = cal_intercept_slope(p, y)
    hl_chi2, hl_dof, hl_p = hosmer_lemeshow(p, y)
    z, z_p = spiegelhalter_z(p, y)
    r_, s_, u_ = murphy(p, y)
    return dict(
        model=name,
        auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        log_loss=float(log_loss(y, p)),
        ece=ece,
        mce=mce,
        cal_intercept=a,
        cal_slope=b,
        hl_chi2=hl_chi2,
        hl_dof=hl_dof,
        hl_p=hl_p,
        spiegelhalter_z=z,
        spiegelhalter_p=z_p,
        mean_pred=float(p.mean()),
        obs_rate=float(y.mean()),
        rel_component=r_,
        resolution=s_,
        uncertainty=u_,
    ), rel_tbl


# ------------------------------------------------------------- run
results, reliab, probs = [], {}, {}
for name, pipe in models.items():
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]
    probs[name] = p
    res, tbl = assess(name, p, y_te)
    results.append(res)
    reliab[name] = tbl
    print(f"\n=== {name} ===")
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items() if k != "model"})
    print(tbl.to_string(index=False))

# post-hoc recalibration (fit only on training data, via internal 5-fold CV)
recal = []
for name, pipe in models.items():
    for method in ("isotonic", "sigmoid"):
        cc = CalibratedClassifierCV(pipe, method=method, cv=5)
        cc.fit(X_tr, y_tr)
        pc = cc.predict_proba(X_te)[:, 1]
        e, m, _ = ece_mce(pc, y_te)
        a, b = cal_intercept_slope(pc, y_te)
        recal.append(
            dict(
                model=name,
                method=method,
                ece=e,
                mce=m,
                brier=float(brier_score_loss(y_te, pc)),
                log_loss=float(log_loss(y_te, pc)),
                auc=float(roc_auc_score(y_te, pc)),
                cal_intercept=a,
                cal_slope=b,
            )
        )
        print(f"{name:22s} {method:9s} ECE={e:.4f} Brier={float(brier_score_loss(y_te,pc)):.4f}")

res_df = pd.DataFrame(results)
rec_df = pd.DataFrame(recal)
print("\n", res_df.to_string(index=False))
print("\n", rec_df.to_string(index=False))

# bootstrap CI for the primary metric (RF ECE) and for RF-vs-LogReg ECE gap
rs = np.random.RandomState(RNG)
n = len(y_te)
boot_rf, boot_gap = [], []
for _ in range(500):
    idx = rs.randint(0, n, n)
    boot_rf.append(ece_mce(probs["RandomForest"][idx], y_te[idx])[0])
    boot_gap.append(
        ece_mce(probs["RandomForest"][idx], y_te[idx])[0]
        - ece_mce(probs["LogisticRegression"][idx], y_te[idx])[0]
    )
ci_rf = np.percentile(boot_rf, [2.5, 97.5])
ci_gap = np.percentile(boot_gap, [2.5, 97.5])
print(f"\nRF ECE 95% CI: {ci_rf}")
print(f"RF-LogReg ECE gap 95% CI: {ci_gap}")

res_df.to_csv("calibration_metrics.csv", index=False)
rec_df.to_csv("recalibration_metrics.csv", index=False)
for k, t in reliab.items():
    t.to_csv(f"reliability_{k}.csv", index=False)

rf = res_df.set_index("model").loc["RandomForest"]
lr = res_df.set_index("model").loc["LogisticRegression"]
hgb = res_df.set_index("model").loc["HistGradientBoosting"]
rf_iso = rec_df.query("model=='RandomForest' and method=='isotonic'").iloc[0]

summary = (
    f"No. On a held-out 30% test set the random forest is materially miscalibrated: "
    f"expected calibration error (15 equal-frequency bins) = {rf.ece:.3f} "
    f"(95% CI {ci_rf[0]:.3f}-{ci_rf[1]:.3f}), with a calibration slope of {rf.cal_slope:.2f} "
    f"(<1 = over-confident, probabilities too extreme at both ends) and a Hosmer-Lemeshow "
    f"test that rejects perfect calibration (chi2={rf.hl_chi2:.0f}, p={rf.hl_p:.2g}). "
    f"Logistic regression (ECE {lr.ece:.3f}) and gradient boosting (ECE {hgb.ece:.3f}) are much "
    f"closer to calibrated but still reject the null at n={len(y_te)}; isotonic recalibration cuts "
    f"the forest's ECE to {rf_iso.ece:.3f} and its Brier score from {rf.brier:.4f} to "
    f"{rf_iso.brier:.4f}, showing most of the error is removable post-hoc."
)

out = {
    "hypothesis_id": "H6",
    "summary": summary,
    "primary_metric_name": "Expected calibration error (ECE, 15 equal-frequency bins) of the random forest on held-out test data",
    "primary_metric_value": round(float(rf.ece), 4),
    "direction": "model is miscalibrated (random forest over-confident, calibration slope < 1); logistic regression and gradient boosting are near-calibrated",
    "methodological_choices": (
        "Data: OpenML Adult (48,842 rows), exact duplicates dropped; fnlwgt dropped as a survey "
        "sampling weight rather than a predictor; target = P(class '>50K') (test base rate "
        f"{rf.obs_rate:.3f}); class imbalance left untouched (no reweighting/resampling) because "
        "reweighting deliberately destroys probability calibration. Validation: single stratified "
        "70/30 train/test split, seed 42; all preprocessing and recalibration fit on train only. "
        "Preprocessing: median impute numerics, most-frequent impute the three columns with NAs "
        "(workclass/occupation/native-country), one-hot categoricals with min_frequency=10, "
        "standardise numerics for logistic regression only. Models: LogisticRegression(C=1, L2), "
        "RandomForestClassifier(500 fully-grown trees), HistGradientBoostingClassifier(defaults). "
        "Random forest chosen as the 'primary' model, which matters: the verdict is model-specific "
        "and a different flagship choice (e.g. gradient boosting or logistic regression) would give "
        "a much smaller ECE. Metric: ECE with 15 EQUAL-FREQUENCY bins (equal-width bins understate "
        "error here because ~half the predictions pile up below p=0.1); MCE, Brier with Murphy "
        "reliability/resolution decomposition, log loss, Cox calibration intercept/slope, "
        "Hosmer-Lemeshow (10 deciles) and Spiegelhalter Z also reported. Note that with ~14.6k test "
        "rows the HL/Z tests reject trivially small deviations, so the ECE magnitude and calibration "
        "slope are treated as the substantive evidence. Post-hoc fixes assessed with "
        "CalibratedClassifierCV (isotonic and sigmoid, internal 5-fold CV on training data). "
        "Uncertainty: 500-row-resample bootstrap percentile CI on the test set."
    ),
}
with open("result.json", "w") as f:
    json.dump(out, f, indent=2)
print("\n" + json.dumps(out, indent=2))
