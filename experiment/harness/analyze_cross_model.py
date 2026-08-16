"""Full cross-model comparison: Sonnet (primary, N=120) vs. Haiku (supplement, N=120),
same 6-hypothesis x 2-arm design. Diversity (using the fixed base-metric canonicalizer
from analyze.py), sign agreement (both arms combined, clearly labeled), and the H1/H4
substantive-agreement coding, replicated for Haiku.
"""
import re
from pathlib import Path

import pandas as pd
import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
SPECIFICITY = {"H1": "abstract", "H2": "concrete", "H3": "abstract",
               "H4": "abstract", "H5": "concrete", "H6": "abstract"}

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
        return "macro_f1"
    if "roc-auc" in s or "roc auc" in s or re.search(r"\bauc\b", s):
        return "roc_auc"
    if "calibration error" in s or re.search(r"\bece\b", s):
        return "ece"
    return "other:" + s[:40]


def h1_comparison_target(name: str) -> frozenset:
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


sonnet = pd.read_csv(EXPERIMENT_DIR / "results_combined.csv")
haiku = pd.read_csv(EXPERIMENT_DIR / "results_haiku_combined.csv")
sonnet["metric_canonical"] = sonnet.apply(lambda r: canonicalize(r["hypothesis_id"], r["primary_metric_name"]), axis=1)
haiku["metric_canonical"] = haiku.apply(lambda r: canonicalize(r["hypothesis_id"], r["primary_metric_name"]), axis=1)
sonnet.to_csv(EXPERIMENT_DIR / "results_with_canonical.csv", index=False)

print("=" * 100)
print("DIVERSITY: unique canonical framings, Sonnet vs Haiku (pooled, out of 20)")
print("=" * 100)
rows = []
for hyp in ["H1", "H2", "H3", "H4", "H5", "H6"]:
    s = sonnet[sonnet["hypothesis_id"] == hyp]
    h = haiku[haiku["hypothesis_id"] == hyp]
    rows.append({
        "hyp": hyp, "specificity": SPECIFICITY[hyp],
        "sonnet_unique_pooled20": s["metric_canonical"].nunique(),
        "haiku_unique_pooled20": h["metric_canonical"].nunique(),
    })
div_df = pd.DataFrame(rows)
print(div_df.to_string(index=False))

print()
print("Per-cell (10-replicate arm) means by specificity, both models:")
percell_rows = []
for hyp in ["H1", "H2", "H3", "H4", "H5", "H6"]:
    for arm in [False, True]:
        s = sonnet[(sonnet["hypothesis_id"] == hyp) & (sonnet["verify_arm"] == arm)]
        h = haiku[(haiku["hypothesis_id"] == hyp) & (haiku["verify_arm"] == arm)]
        percell_rows.append({"hyp": hyp, "specificity": SPECIFICITY[hyp],
                              "sonnet_unique": s["metric_canonical"].nunique(),
                              "haiku_unique": h["metric_canonical"].nunique()})
pc = pd.DataFrame(percell_rows)
print(pc.groupby("specificity")[["sonnet_unique", "haiku_unique"]].mean().to_string())

print()
print("=" * 100)
print("DIRECTIONAL AGREEMENT: H2 and H5, BOTH ARMS COMBINED (n=20 each)")
print("=" * 100)
for hyp in ["H2", "H5"]:
    for name, d in [("Sonnet", sonnet[sonnet["hypothesis_id"] == hyp]),
                     ("Haiku", haiku[haiku["hypothesis_id"] == hyp])]:
        vals = d["primary_metric_value"].dropna()
        pos = (vals > 0).sum()
        neg = (vals < 0).sum()
        print(f"{hyp} {name}: n={len(vals)}, mean={vals.mean():.4f}, pos={pos}, neg={neg}, "
              f"majority={max(pos,neg)/len(vals):.0%}")

print()
print("H5 direction-field check (both models, both arms, n=40): does EITHER model ever")
print("report a meaningful SMOTE effect (i.e. disagree with 'no meaningful change')?")
for name, d in [("Sonnet", sonnet[sonnet["hypothesis_id"] == "H5"]),
                ("Haiku", haiku[haiku["hypothesis_id"] == "H5"])]:
    no_change = d["direction"].str.contains("no meaningful|not meaningfully|negligible|does not exceed|"
                                             "no practically significant|no substantial|within.*threshold|"
                                             "below.*threshold|does not move|does NOT produce", case=False, na=False, regex=True)
    print(f"  {name}: {no_change.sum()}/{len(d)} state no-meaningful-change")

print()
print("=" * 100)
print("H1 SUBSTANTIVE CHECK: does a tree ensemble rank first? (Sonnet vs Haiku)")
print("=" * 100)
tree_family = ['gradientboosting', 'histgradientboosting', 'randomforest', 'gradient boosting',
               'hist gradient boosting', 'random forest']


def h1_code(direction):
    d = str(direction).lower()
    first_chunk = d.split('>')[0]
    return 'tree_ensemble_wins' if any(t in first_chunk for t in tree_family) else 'other'


for name, d in [("Sonnet", sonnet[sonnet["hypothesis_id"] == "H1"]),
                ("Haiku", haiku[haiku["hypothesis_id"] == "H1"])]:
    dd = d.copy()
    dd["code"] = dd["direction"].apply(h1_code)
    rate = (dd["code"] == "tree_ensemble_wins").mean()
    print(f"{name}: {(dd['code']=='tree_ensemble_wins').sum()}/{len(dd)} = {rate:.0%}")
