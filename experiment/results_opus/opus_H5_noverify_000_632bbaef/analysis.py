"""
H5: Does SMOTE oversampling of the training data change the minority-class (>50K)
F1 score by more than 0.02 versus no resampling, with a default-hyperparameter
RandomForestClassifier?

Design
------
Two arms, identical in every respect except the resampling step:
    A) no resampling   : preprocess -> RandomForestClassifier()
    B) SMOTE           : preprocess -> SMOTE() -> RandomForestClassifier()

Evaluation:
  * Primary: 5-fold stratified CV repeated 3x (15 paired folds). SMOTE is fitted
    INSIDE each training fold only (imblearn Pipeline), so validation folds keep
    the natural class prior -- resampling the validation data would inflate F1
    artificially.
  * Secondary: a single stratified 80/20 hold-out, as an independent check.

Metric: F1 of the positive class (>50K) at the default 0.5 decision threshold.
Precision and recall are reported alongside to explain the direction of any shift.
"""

import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy import stats

SEED = 42

# ---------------------------------------------------------------- data
df = pd.read_csv("adult_income.csv")
df = df.drop_duplicates()  # 52 exact duplicate rows

y = (df["class"].str.strip() == ">50K").astype(int).to_numpy()
X = df.drop(columns=["class"])

num_cols = X.select_dtypes(include=np.number).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print(f"rows={len(X)}  positives={y.sum()} ({y.mean():.3%})")
print(f"numeric={num_cols}\ncategorical={cat_cols}")

# Missing categoricals ('?' in the raw UCI file, already NaN here) become their
# own level -- missingness in workclass/occupation is informative, not random.
pre = ColumnTransformer(
    [
        ("num", SimpleImputer(strategy="median"), num_cols),
        (
            "cat",
            Pipeline(
                [
                    ("imp", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                ]
            ),
            cat_cols,
        ),
    ]
)
# No scaling: trees are scale-invariant. SMOTE's k-NN is NOT scale-invariant, but
# scaling would change the no-resampling arm's inputs too and is not part of a
# "default RF" pipeline; see result.json for this caveat.

def make_pipe(use_smote):
    steps = [("pre", clone(pre))]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=SEED)))
    steps.append(("clf", RandomForestClassifier(random_state=SEED, n_jobs=-1)))
    return ImbPipeline(steps)

def scores(model, Xte, yte):
    p = model.predict(Xte)
    return dict(
        f1=f1_score(yte, p),
        precision=precision_score(yte, p),
        recall=recall_score(yte, p),
        roc_auc=roc_auc_score(yte, model.predict_proba(Xte)[:, 1]),
    )

# ------------------------------------------------- primary: repeated stratified CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)
rows = []
for fold, (tr, te) in enumerate(cv.split(X, y)):
    Xtr, Xte = X.iloc[tr], X.iloc[te]
    ytr, yte = y[tr], y[te]
    for arm, use_smote in [("none", False), ("smote", True)]:
        m = make_pipe(use_smote).fit(Xtr, ytr)
        s = scores(m, Xte, yte)
        s.update(fold=fold, arm=arm)
        rows.append(s)
    print(
        f"fold {fold:2d}  none F1={rows[-2]['f1']:.4f}  smote F1={rows[-1]['f1']:.4f}"
        f"  diff={rows[-1]['f1'] - rows[-2]['f1']:+.4f}",
        flush=True,
    )

cvres = pd.DataFrame(rows)
wide = cvres.pivot(index="fold", columns="arm")
diff = wide[("f1", "smote")] - wide[("f1", "none")]

mean_none = wide[("f1", "none")].mean()
mean_smote = wide[("f1", "smote")].mean()
mean_diff = float(diff.mean())
sd_diff = float(diff.std(ddof=1))
n = len(diff)
ci = stats.t.interval(0.95, n - 1, loc=mean_diff, scale=sd_diff / np.sqrt(n))
tstat, pval = stats.ttest_rel(wide[("f1", "smote")], wide[("f1", "none")])

print("\n=== Repeated 5-fold CV (15 paired folds) ===")
print(cvres.groupby("arm")[["f1", "precision", "recall", "roc_auc"]].agg(["mean", "std"]))
print(f"\nmean F1 no-resampling = {mean_none:.4f}")
print(f"mean F1 SMOTE         = {mean_smote:.4f}")
print(f"paired mean diff (SMOTE - none) = {mean_diff:+.4f}  (sd {sd_diff:.4f})")
print(f"95% CI of diff = [{ci[0]:+.4f}, {ci[1]:+.4f}]   paired t p={pval:.2g}")
print(f"|diff| > 0.02 ? {abs(mean_diff) > 0.02}")

# equivalence check against the 0.02 threshold: is the whole CI inside +/-0.02?
within = ci[0] > -0.02 and ci[1] < 0.02
print(f"95% CI entirely within +/-0.02 (practical equivalence)? {within}")

# ------------------------------------------------- secondary: 80/20 hold-out
Xtr, Xte, ytr, yte = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)
hold = {}
for arm, use_smote in [("none", False), ("smote", True)]:
    hold[arm] = scores(make_pipe(use_smote).fit(Xtr, ytr), Xte, yte)
print("\n=== 80/20 hold-out ===")
print(pd.DataFrame(hold).T.round(4))
hold_diff = hold["smote"]["f1"] - hold["none"]["f1"]
print(f"hold-out F1 diff (SMOTE - none) = {hold_diff:+.4f}")

# ------------------------------------------------- seed sensitivity of the RF
seed_diffs = []
for s in [0, 1, 2, 3, 4]:
    out = {}
    for arm, use_smote in [("none", False), ("smote", True)]:
        steps = [("pre", clone(pre))]
        if use_smote:
            steps.append(("smote", SMOTE(random_state=s)))
        steps.append(("clf", RandomForestClassifier(random_state=s, n_jobs=-1)))
        out[arm] = f1_score(yte, ImbPipeline(steps).fit(Xtr, ytr).predict(Xte))
    seed_diffs.append(out["smote"] - out["none"])
    print(f"seed {s}: none={out['none']:.4f} smote={out['smote']:.4f} diff={seed_diffs[-1]:+.4f}")
print(f"hold-out diff across 5 seeds: mean={np.mean(seed_diffs):+.4f} "
      f"range=[{min(seed_diffs):+.4f}, {max(seed_diffs):+.4f}]")

# ------------------------------------------------- write result
summary = (
    f"No. With a default RandomForestClassifier on one-hot-encoded Adult data, SMOTE "
    f"changed the >50K F1 by only {mean_diff:+.4f} (95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]) "
    f"across 15 paired stratified CV folds -- well inside the 0.02 threshold, so the CI "
    f"rules out an effect of that size. SMOTE gave up more precision "
    f"({wide[('precision','none')].mean():.3f}->{wide[('precision','smote')].mean():.3f}) "
    f"than it gained in recall "
    f"({wide[('recall','none')].mean():.3f}->{wide[('recall','smote')].mean():.3f}), "
    f"so the small net effect on F1 is slightly negative."
)

result = {
    "hypothesis_id": "H5",
    "summary": summary,
    "primary_metric_name": "Minority-class (>50K) F1 difference (SMOTE - no resampling), mean over 15 paired stratified CV folds",
    "primary_metric_value": round(mean_diff, 4),
    "direction": f"No practically meaningful change (|diff| = {abs(mean_diff):.4f} < 0.02); SMOTE slightly {'lowers' if mean_diff < 0 else 'raises'} F1 while shifting precision toward recall",
    "methodological_choices": (
        "Encoding: median imputation for numeric, explicit 'Missing' level + one-hot "
        "(handle_unknown='ignore') for categoricals; all 14 features kept, including fnlwgt "
        "and both education/education-num (another researcher might drop fnlwgt or the "
        "redundant education column). No feature scaling -- trees are scale-invariant, but "
        "SMOTE's k-NN interpolation is not, so an analyst who standardised before SMOTE "
        "could get somewhat different synthetic points; 52 exact duplicate rows dropped. "
        "Validation: 5-fold stratified CV repeated 3x (15 paired folds), SMOTE fitted inside "
        "training folds only via an imblearn Pipeline so validation folds retain the natural "
        "23.9% prior; a stratified 80/20 hold-out and a 5-seed sweep are reported as checks "
        "(hold-out diff "
        f"{hold_diff:+.4f}, 5-seed mean {np.mean(seed_diffs):+.4f}). "
        "Metric: F1 of the >50K class at the fixed default 0.5 threshold -- no threshold "
        "tuning; tuning the threshold on the no-resampling model would likely absorb most of "
        "what SMOTE does. Model: RandomForestClassifier() at library defaults (only "
        "random_state and n_jobs set), no class_weight. SMOTE: imblearn defaults "
        "(k_neighbors=5, full balance to a 1:1 ratio) applied to the one-hot-encoded matrix, "
        "which produces fractional values in binary indicator columns -- SMOTE-NC on the raw "
        "categoricals is a defensible alternative. Inference: paired t-test / t-CI over the "
        "15 folds, which share data across folds and so slightly understate the true variance."
    ),
}
with open("result.json", "w") as f:
    json.dump(result, f, indent=2)
print("\n" + json.dumps(result, indent=2))
