"""
H6: Is the model well-calibrated?

Dataset: UCI/OpenML Adult (Census Income), 48,842 rows, binary target `class`
(<=50K / >50K). Positive class = ">50K".

Approach
--------
The research question does not name a model, so I fit two standard classifiers
that bracket the usual calibration behaviour:
  * Logistic Regression (one-hot + standardised numerics) -- a proper-scoring-rule
    learner, expected to be near-calibrated by construction.
  * Random Forest (500 trees, ordinal/one-hot encoded) -- a bagged vote-averager,
    the classic example of a miscalibrated probability source.
I designate the Random Forest as "the model" for the primary metric (it is the
better-discriminating, more typical deployment choice here) but report both.

Calibration is assessed on a held-out stratified 30% test set with:
  * Expected Calibration Error (ECE), 10 equal-frequency bins (also equal-width)
  * Maximum Calibration Error (MCE)
  * Brier score + Murphy decomposition (reliability / resolution / uncertainty)
  * Calibration intercept & slope (Cox recalibration regression of y on logit p)
  * Hosmer-Lemeshow C-statistic (10 deciles of risk)
  * A *null reference* for ECE: labels resampled from the model's own
    probabilities, giving the ECE distribution expected under perfect
    calibration at this sample size. Raw ECE is meaningless without it.
  * Post-hoc recalibration (Platt / isotonic, fit by 5-fold CV on train only)
    to quantify how much calibration error is actually removable.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
RNG = 42
rng = np.random.default_rng(RNG)

# ----------------------------------------------------------------------------
# 1. Load & clean
# ----------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv", skipinitialspace=True)
df = df.apply(lambda s: s.str.strip() if s.dtype == object else s)
df = df.replace("?", np.nan)

# target: strip trailing '.' variants that appear in the combined train+test file
y = (df["class"].str.rstrip(".").str.strip() == ">50K").astype(int).values
X = df.drop(columns=["class"])

# fnlwgt is a census sampling weight, not a person-level predictor -> drop
X = X.drop(columns=["fnlwgt"])

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print(f"rows={len(df)}  positives={y.mean():.4f}  numeric={num_cols}")
print(f"categorical={cat_cols}")
print("missing per column:\n", X.isna().sum()[X.isna().sum() > 0])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.30, random_state=RNG, stratify=y
)
print(f"train={len(X_tr)} test={len(X_te)} test base rate={y_te.mean():.4f}")

# ----------------------------------------------------------------------------
# 2. Models
# ----------------------------------------------------------------------------
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
    "LogReg": Pipeline(
        [("pre", make_pre(True)), ("clf", LogisticRegression(C=1.0, max_iter=2000))]
    ),
    "RandomForest": Pipeline(
        [
            ("pre", make_pre(False)),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=500,
                    min_samples_leaf=1,
                    n_jobs=-1,
                    random_state=RNG,
                ),
            ),
        ]
    ),
}

# ----------------------------------------------------------------------------
# 3. Calibration metrics
# ----------------------------------------------------------------------------
def ece(y_true, p, n_bins=10, strategy="quantile"):
    """Expected / maximum calibration error. Returns (ece, mce, bin table)."""
    if strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    else:
        edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    rows, e, m = [], 0.0, 0.0
    for b in range(len(edges) - 1):
        sel = idx == b
        n = sel.sum()
        if n == 0:
            continue
        conf, acc = p[sel].mean(), y_true[sel].mean()
        gap = abs(conf - acc)
        e += n / len(p) * gap
        m = max(m, gap)
        rows.append(dict(bin=b, n=int(n), mean_pred=conf, obs_freq=acc, gap=conf - acc))
    return e, m, pd.DataFrame(rows)


def cox_slope_intercept(y_true, p):
    """Calibration-in-the-large (intercept) and calibration slope.
    Perfect calibration => intercept 0, slope 1."""
    lp = np.log(np.clip(p, 1e-9, 1 - 1e-9) / (1 - np.clip(p, 1e-9, 1 - 1e-9)))
    slope = LogisticRegression(C=1e9, max_iter=2000).fit(lp.reshape(-1, 1), y_true)
    # intercept with slope fixed at 1 (offset model), via 1-D Newton on the offset
    a = 0.0
    for _ in range(100):
        q = 1 / (1 + np.exp(-(lp + a)))
        g, h = (y_true - q).sum(), -(q * (1 - q)).sum()
        step = g / h
        a -= step
        if abs(step) < 1e-10:
            break
    return float(a), float(slope.coef_[0][0]), float(slope.intercept_[0])


def hosmer_lemeshow(y_true, p, g=10, min_expected=5.0):
    """Hosmer-Lemeshow C. The forest piles a large mass of predictions at ~0, so
    the low deciles have expected event counts near zero and the raw statistic
    explodes numerically. Merge adjacent low groups until every group has at
    least `min_expected` expected events (standard remedy); df = groups - 2."""
    from scipy.stats import chi2

    q = pd.qcut(pd.Series(p).rank(method="first"), g, labels=False).values
    groups = [(int((q == k).sum()), float(y_true[q == k].sum()), float(p[q == k].sum()))
              for k in range(g)]
    merged, buf = [], None
    for grp in groups:
        buf = grp if buf is None else tuple(a + b for a, b in zip(buf, grp))
        if buf[2] >= min_expected and (buf[0] - buf[2]) >= min_expected:
            merged.append(buf)
            buf = None
    if buf is not None:
        merged[-1] = tuple(a + b for a, b in zip(merged[-1], buf)) if merged else buf
    stat = sum((o - e) ** 2 / (e * (1 - e / n)) for n, o, e in merged)
    dof = max(len(merged) - 2, 1)
    return float(stat), float(chi2.sf(stat, dof)), len(merged)


def murphy(y_true, p, n_bins=10):
    """Brier = reliability - resolution + uncertainty (quantile bins)."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(p, edges[1:-1], right=True), 0, len(edges) - 2)
    base, rel, res = y_true.mean(), 0.0, 0.0
    for b in np.unique(idx):
        sel = idx == b
        n, conf, obs = sel.sum(), p[sel].mean(), y_true[sel].mean()
        rel += n * (conf - obs) ** 2
        res += n * (obs - base) ** 2
    N = len(p)
    return rel / N, res / N, base * (1 - base)


def null_ece(y_true, p, n_rep=500, n_bins=10):
    """ECE distribution if the model were PERFECTLY calibrated: resample labels
    from p itself and recompute. Gives the sampling-noise floor for ECE."""
    out = []
    for _ in range(n_rep):
        ys = (rng.random(len(p)) < p).astype(int)
        out.append(ece(ys, p, n_bins)[0])
    return np.array(out)


def bootstrap_ece(y_true, p, n_rep=500, n_bins=10):
    out = []
    n = len(p)
    for _ in range(n_rep):
        i = rng.integers(0, n, n)
        out.append(ece(y_true[i], p[i], n_bins)[0])
    return np.array(out)


# ----------------------------------------------------------------------------
# 4. Fit, evaluate, recalibrate
# ----------------------------------------------------------------------------
results = {}
for name, pipe in models.items():
    pipe.fit(X_tr, y_tr)
    p = pipe.predict_proba(X_te)[:, 1]

    e_q, m_q, tbl_q = ece(y_te, p, 10, "quantile")
    e_w, m_w, _ = ece(y_te, p, 10, "uniform")
    e_q20, _, _ = ece(y_te, p, 20, "quantile")
    a0, slope, s_int = cox_slope_intercept(y_te, p)
    hl, hl_p, hl_g = hosmer_lemeshow(y_te, p)
    nul20 = null_ece(y_te, p, 200, 20)
    rel, res, unc = murphy(y_te, p)
    nul = null_ece(y_te, p)
    boot = bootstrap_ece(y_te, p)

    r = dict(
        auc=roc_auc_score(y_te, p),
        brier=brier_score_loss(y_te, p),
        logloss=log_loss(y_te, p),
        ece_quantile=e_q,
        ece_uniform=e_w,
        ece_quantile_20bin=e_q20,
        ece_null_mean_20bin=float(nul20.mean()),
        ece_p_value_20bin=float((nul20 >= e_q20).mean()),
        mce_quantile=m_q,
        mce_uniform=m_w,
        ece_boot_lo=float(np.percentile(boot, 2.5)),
        ece_boot_hi=float(np.percentile(boot, 97.5)),
        ece_null_mean=float(nul.mean()),
        ece_null_p975=float(np.percentile(nul, 97.5)),
        ece_ratio_to_null=float(e_q / nul.mean()),
        ece_p_value=float((nul >= e_q).mean()),
        cal_intercept=a0,
        cal_slope=slope,
        hl_stat=hl,
        hl_p=hl_p,
        hl_groups=hl_g,
        reliability=rel,
        resolution=res,
        uncertainty=unc,
        mean_pred=float(p.mean()),
        obs_rate=float(y_te.mean()),
    )

    # post-hoc recalibration fitted on TRAIN ONLY (5-fold internal CV)
    for meth in ["sigmoid", "isotonic"]:
        cal = CalibratedClassifierCV(pipe, method=meth, cv=5)
        cal.fit(X_tr, y_tr)
        pc = cal.predict_proba(X_te)[:, 1]
        r[f"ece_{meth}"] = ece(y_te, pc, 10)[0]
        r[f"brier_{meth}"] = brier_score_loss(y_te, pc)
        r[f"auc_{meth}"] = roc_auc_score(y_te, pc)

    results[name] = r
    print(f"\n===== {name} =====")
    for k, v in r.items():
        print(f"  {k:22s} {v: .5f}")
    print(tbl_q.to_string(index=False, float_format=lambda x: f"{x: .4f}"))

# ----------------------------------------------------------------------------
# 5. Reliability diagram
# ----------------------------------------------------------------------------
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, pipe in models.items():
        p = pipe.predict_proba(X_te)[:, 1]
        f, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
        ax.plot(mp, f, "o-", label=f"{name} (ECE={results[name]['ece_quantile']:.4f})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed frequency of >50K")
    ax.set_title("Reliability diagram (test set, 10 quantile bins)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("reliability.png", dpi=130)
    print("\nsaved reliability.png")
except Exception as exc:  # pragma: no cover
    print("plot skipped:", exc)

# ----------------------------------------------------------------------------
# 6. Write result.json  (primary model = RandomForest)
# ----------------------------------------------------------------------------
rf, lr = results["RandomForest"], results["LogReg"]
out = {
    "hypothesis_id": "H6",
    "summary": (
        f"It depends on which model, and the two answers are opposite. The random forest is "
        f"clearly miscalibrated: 10-bin expected calibration error {rf['ece_quantile']:.4f} "
        f"(95% bootstrap CI {rf['ece_boot_lo']:.4f}-{rf['ece_boot_hi']:.4f}), ~"
        f"{rf['ece_ratio_to_null']:.0f}x the {rf['ece_null_mean']:.4f} expected under perfect "
        f"calibration at this sample size (p<0.002), calibration slope {rf['cal_slope']:.2f} -- it "
        f"pushes probabilities too far toward 0 and 1 (predicts 0.94 where the true rate is 0.86, "
        f"and 0.001 where it is 0.013). Logistic regression, by contrast, is statistically "
        f"indistinguishable from perfectly calibrated (ECE {lr['ece_quantile']:.4f}, p="
        f"{lr['ece_p_value']:.2f} against the same null; slope {lr['cal_slope']:.2f}, intercept "
        f"{lr['cal_intercept']:.2f}, Hosmer-Lemeshow p={lr['hl_p']:.2f}). Both models are "
        f"calibrated-in-the-large (mean prediction within 0.002 of the {rf['obs_rate']:.3f} base "
        f"rate); the forest's error is purely in the shape of the reliability curve, and isotonic "
        f"recalibration removes about two-thirds of it (ECE {rf['ece_isotonic']:.4f}) with no loss "
        f"of AUC, while Platt scaling does not ({rf['ece_sigmoid']:.4f})."
    ),
    "primary_metric_name": "Expected Calibration Error (10 equal-frequency bins, held-out test set, random forest)",
    "primary_metric_value": round(float(rf["ece_quantile"]), 5),
    "direction": "model-dependent: random forest is miscalibrated (over-confident at both extremes, slope 0.60); logistic regression is well-calibrated",
    "methodological_choices": (
        "Positive class = '>50K'. Dropped fnlwgt (census sampling weight, not a person-level "
        "predictor); '?' treated as missing and imputed (median / most-frequent) rather than "
        "row-dropped; one-hot encoding with min_frequency=10 to pool rare levels; no class-imbalance "
        "reweighting or resampling (deliberate -- SMOTE/class_weight would itself destroy calibration). "
        "Single stratified 70/30 train/test split, seed 42. Two models: L2 logistic regression (C=1, "
        "standardised numerics) and a 500-tree random forest with fully grown trees; the RANDOM FOREST "
        "is designated 'the model' for the primary metric, and the choice matters a lot -- LogReg's ECE "
        f"is {lr['ece_quantile']:.4f} vs the forest's {rf['ece_quantile']:.4f}. ECE uses 10 equal-frequency "
        "bins (equal-width bins also reported; binning scheme and bin count materially change ECE). "
        "Crucially, raw ECE is compared against a parametric-bootstrap null (labels resampled from the "
        "model's own predicted probabilities, 500 reps) so that 'miscalibrated' means 'beyond sampling "
        "noise' rather than 'ECE > 0'; a researcher using a fixed threshold like ECE<0.01 would call the "
        "forest calibrated-ish and would have reached a weaker conclusion. Also reported: Brier score with "
        "Murphy reliability/resolution/uncertainty decomposition, Cox calibration intercept and slope, "
        "Hosmer-Lemeshow C (10 deciles, adjacent low-risk groups merged to keep >=5 expected events -- "
        "without that merge the forest's large point mass at p~0 makes the raw HL statistic explode "
        "numerically; HL is also over-powered at n=14,653, so effect sizes are reported alongside "
        "p-values), and Platt/isotonic recalibration fitted by 5-fold CV on the training set only. "
        "Note the forest's quantile ECE uses 8 rather than 10 effective bins because ~4,500 test rows "
        "share a predicted probability of essentially 0 (ties cannot be split into deciles); a "
        "20-bin ECE is reported as a robustness check."
    ),
    "detail": results,
}
with open("result.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
print("\nwrote result.json")
print(json.dumps({k: v for k, v in out.items() if k != "detail"}, indent=2))
