"""Full cross-model comparison: Sonnet (primary, N=120) vs. Haiku (supplement, N=120),
same 6-hypothesis x 2-arm design. Diversity (using the shared canonicalizer in
canonicalize.py), sign agreement (both arms combined, clearly labeled), and the H1/H4
substantive-agreement coding, replicated for Haiku.
"""
import re
from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonicalize import canonicalize

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
SPECIFICITY = {"H1": "abstract", "H2": "concrete", "H3": "abstract",
               "H4": "abstract", "H5": "concrete", "H6": "abstract"}

sonnet = pd.read_csv(EXPERIMENT_DIR / "results_combined.csv")
haiku = pd.read_csv(EXPERIMENT_DIR / "results_haiku_combined.csv")
sonnet["metric_canonical"] = sonnet.apply(lambda r: canonicalize(r["hypothesis_id"], r["primary_metric_name"], r.get("methodological_choices")), axis=1)
haiku["metric_canonical"] = haiku.apply(lambda r: canonicalize(r["hypothesis_id"], r["primary_metric_name"], r.get("methodological_choices")), axis=1)
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
