"""Core analysis: measure-choice diversity, numeric dispersion, and the verification-arm
comparison, per hypothesis.

Canonicalization note (fixed 2026-08-16): the original canonicalize() collapsed only
CV/fold/test-set wording, which (a) had a regex bug that deleted entire H1 names starting
with "CV " down to the empty string, and (b) still left model-specific and repeat/seed
qualifiers uncollapsed, so replicates using the identical measure (e.g. H5's "F1
difference, SMOTE - no resampling" or H6's "Expected Calibration Error, 10-bin") were
counted as distinct "framings" purely because of wording. The replacement below extracts
(1) the base statistical measure (ROC-AUC / balanced accuracy / macro-F1 / minority-class
F1 / ECE / feature-importance sub-type) and, for H1 only, (2) which pair of model families
is being compared, since H1's question ("does model family affect performance?") is not
well-defined without choosing that pair -- unlike H3's "top feature" or H6's "which model
to calibrate", which are outputs of a fixed method rather than a chosen operationalization,
and are already tracked separately (Part 2, Table 3/4: model-class usage).
"""
import re
from pathlib import Path

import pandas as pd
import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 160)

df = pd.read_csv(EXPERIMENT_DIR / "results_combined.csv")

SPECIFICITY = {
    "H1": "abstract", "H2": "concrete", "H3": "abstract",
    "H4": "abstract", "H5": "concrete", "H6": "abstract",
}
df["specificity"] = df["hypothesis_id"].map(SPECIFICITY)

MODEL_ALIASES = {
    "histgradientboosting": "hgb", "hist gradient boosting": "hgb", "hgb": "hgb",
    "gradientboosting": "gb", "gradient boosting": "gb", "gbm": "gb",
    "randomforest": "rf", "random forest": "rf", "rf": "rf",
    "logisticregression": "logreg", "logistic regression": "logreg", "logreg": "logreg",
    "knn": "knn", "k-nearest neighbors": "knn",
    "gaussiannb": "nb", "naive bayes": "nb",
    "best tree ensemble": "best_tree_ensemble",
    "best": "best", "worst": "worst", "max": "best", "min": "worst",
}


def base_metric(name: str) -> str:
    """The underlying statistical quantity, independent of wording, CV/split details, or
    which specific model/feature is named."""
    s = name.lower()
    if "feature importance" in s or "permutation importance" in s or "decrease in impurity" in s or "feature_importance" in s:
        s_noseed = re.sub(r"averaged? (over|across) \d+ (seeds?|reruns?|folds?|repeats?)", "", s)
        if any(k in s_noseed for k in ("consensus", "aggregated", "combined")) or \
           ("average" in s_noseed and "normalized" in s_noseed):
            return "feat_importance:consensus_multi_method"
        if "permutation" in s:
            return "feat_importance:permutation"
        if "impurity" in s or "gini" in s:
            return "feat_importance:impurity_mdi"
        return "feat_importance:unspecified"
    if "balanced accuracy" in s or "balanced-accuracy" in s:
        return "balanced_accuracy"
    if "macro-f1" in s or "macro f1" in s or "f1-macro" in s or "f1 macro" in s:
        return "macro_f1"
    if re.search(r"\bf1\b", s) and ">50k" in s.replace(" ", ""):
        return "f1_minority"
    if re.search(r"\bf1\b", s):
        return "macro_f1"  # bare "F1" in H4 context means macro-F1
    if "roc-auc" in s or "roc auc" in s or re.search(r"\bauc\b", s):
        return "roc_auc"
    if "calibration error" in s or re.search(r"\bece\b", s):
        return "ece"
    return "other:" + s[:40]


def h1_comparison_target(name: str) -> frozenset:
    """Which pair of model families H1 compares (order-independent)."""
    s = name.lower()
    m = re.search(r"\(([^)]*)\)", s)
    inside = m.group(1) if m else s
    inside = re.split(r",", inside)[0]
    parts = re.split(r"\s*-\s*|\sminus\s", inside)
    norm = [MODEL_ALIASES.get(p.strip()) for p in parts]
    norm = [p for p in norm if p]
    return frozenset(norm) if norm else frozenset(["unresolved"])


def canonicalize(hyp: str, name: str) -> str:
    bm = base_metric(name)
    if hyp == "H1":
        return f"{bm}|{'+'.join(sorted(h1_comparison_target(name)))}"
    return bm


df["metric_canonical"] = df.apply(lambda r: canonicalize(r["hypothesis_id"], r["primary_metric_name"]), axis=1)

print("=" * 100)
print("MEASURE-CHOICE DIVERSITY (unique canonicalized metric framings per cell)")
print("=" * 100)
diversity_rows = []
for (hyp, arm), sub in df.groupby(["hypothesis_id", "verify_arm"]):
    n = len(sub)
    n_unique_raw = sub["primary_metric_name"].nunique()
    n_unique_canon = sub["metric_canonical"].nunique()
    top = sub["metric_canonical"].value_counts().iloc[0]
    diversity_rows.append({
        "hypothesis_id": hyp, "specificity": SPECIFICITY[hyp], "verify_arm": arm, "n": n,
        "unique_raw_names": n_unique_raw, "unique_canonical": n_unique_canon,
        "modal_framing_share": round(top / n, 4),
    })
div_df = pd.DataFrame(diversity_rows).sort_values(["specificity", "hypothesis_id", "verify_arm"])
print(div_df.to_string(index=False))

print()
print("=" * 100)
print("POOLED (both arms, 20 replicates per hypothesis)")
print("=" * 100)
for hyp, sub in df.groupby("hypothesis_id"):
    print(hyp, SPECIFICITY[hyp], "unique_canonical_pooled=", sub["metric_canonical"].nunique())

print()
print("=" * 100)
print("SUMMARY: abstract vs concrete, mean per (hypothesis, arm) cell of 10")
print("=" * 100)
summary = df.groupby("specificity").apply(
    lambda s: pd.Series({
        "n": len(s),
        "mean_unique_canonical_per_cell": div_df[div_df["specificity"] == s.name]["unique_canonical"].mean(),
        "mean_modal_share": div_df[div_df["specificity"] == s.name]["modal_framing_share"].mean(),
    }),
    include_groups=False,
)
print(summary.to_string())
print("diversity ratio (abstract/concrete):", round(summary.loc["abstract", "mean_unique_canonical_per_cell"] / summary.loc["concrete", "mean_unique_canonical_per_cell"], 2))
print("modal-share ratio (concrete/abstract):", round(summary.loc["concrete", "mean_modal_share"] / summary.loc["abstract", "mean_modal_share"], 2))

df.to_csv(EXPERIMENT_DIR / "results_with_canonical.csv", index=False)
print("\nSaved results_with_canonical.csv")
