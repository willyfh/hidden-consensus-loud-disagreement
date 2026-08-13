"""Full cross-model-family comparison: Sonnet (primary, N=120) vs. Haiku (supplement,
N=120), same 6-hypothesis x 2-arm design. Diversity, sign agreement, and the H1/H4
substantive-agreement coding, replicated for Haiku.
"""
import re
from pathlib import Path

import pandas as pd
import numpy as np

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
SPECIFICITY = {"H1": "abstract", "H2": "concrete", "H3": "abstract",
               "H4": "abstract", "H5": "concrete", "H6": "abstract"}


def canonicalize(name: str) -> str:
    s = name.lower()
    s = re.sub(r"\(.*?repeat.*?\)", "", s)
    s = re.sub(r"mean (over|across).*?(fold|cv|split|seed|repeat)s?.*", "", s)
    s = re.sub(r",?\s*(5|10|20|25)[\s-]?fold.*", "", s)
    s = re.sub(r",?\s*(stratified )?cv.*", "", s)
    s = re.sub(r",?\s*(held[- ]?out )?test set.*", "", s)
    s = re.sub(r",?\s*single held[- ]?out.*", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,")
    return s


sonnet = pd.read_csv(EXPERIMENT_DIR / "results_with_canonical.csv")
haiku = pd.read_csv(EXPERIMENT_DIR / "results_haiku_combined.csv")
haiku["metric_canonical"] = haiku["primary_metric_name"].apply(canonicalize)

print("=" * 100)
print("DIVERSITY: unique canonical framings per cell, Sonnet vs Haiku (out of 10)")
print("=" * 100)
rows = []
for hyp in ["H1", "H2", "H3", "H4", "H5", "H6"]:
    for arm in [False, True]:
        s = sonnet[(sonnet["hypothesis_id"] == hyp) & (sonnet["verify_arm"] == arm)]
        h = haiku[(haiku["hypothesis_id"] == hyp) & (haiku["verify_arm"] == arm)]
        rows.append({
            "hyp": hyp, "specificity": SPECIFICITY[hyp], "arm": "verify" if arm else "noverify",
            "sonnet_unique": s["metric_canonical"].nunique(),
            "haiku_unique": h["metric_canonical"].nunique(),
        })
div_df = pd.DataFrame(rows)
print(div_df.to_string(index=False))

print()
summary = div_df.groupby("specificity")[["sonnet_unique", "haiku_unique"]].mean()
print("Mean unique framings by specificity (both models):")
print(summary.to_string())

print()
print("=" * 100)
print("DIRECTIONAL AGREEMENT: H2 and H5 (concrete, numerically comparable), no-verify arm")
print("=" * 100)
for hyp in ["H2", "H5"]:
    for name, df in [("Sonnet", sonnet[(sonnet["hypothesis_id"] == hyp) & (sonnet["verify_arm"] == False)]),
                      ("Haiku", haiku[(haiku["hypothesis_id"] == hyp) & (haiku["verify_arm"] == False)])]:
        vals = df["primary_metric_value"].dropna()
        pos = (vals > 0).sum()
        neg = (vals < 0).sum()
        print(f"{hyp} {name}: n={len(vals)}, mean={vals.mean():.4f}, pos={pos}, neg={neg}, "
              f"majority={max(pos,neg)/len(vals):.0%}")

print()
print("=" * 100)
print("H1 SUBSTANTIVE CHECK (Haiku): does a tree ensemble rank first?")
print("=" * 100)
tree_family = ['gradientboosting', 'histgradientboosting', 'randomforest', 'gradient boosting',
               'hist gradient boosting', 'random forest']


def h1_code(direction):
    d = str(direction).lower()
    first_chunk = d.split('>')[0]
    return 'tree_ensemble_wins' if any(t in first_chunk for t in tree_family) else 'other'


h1_haiku = haiku[haiku["hypothesis_id"] == "H1"].copy()
h1_haiku["code"] = h1_haiku["direction"].apply(h1_code)
print(h1_haiku["code"].value_counts())
print(f"agreement rate: {(h1_haiku['code']=='tree_ensemble_wins').mean():.0%}")
