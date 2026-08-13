"""Core analysis: measure-choice diversity, numeric dispersion, and the verification-arm
comparison, per hypothesis.
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


def canonicalize(name: str) -> str:
    """Strip generic CV/aggregation phrasing so trivial rewordings collapse together,
    while genuinely different comparisons (different model pair, different feature,
    different base metric) remain distinct."""
    s = name.lower()
    s = re.sub(r"\(.*?repeat.*?\)", "", s)
    s = re.sub(r"mean (over|across).*?(fold|cv|split|seed|repeat)s?.*", "", s)
    s = re.sub(r",?\s*(5|10|20|25)[\s-]?fold.*", "", s)
    s = re.sub(r",?\s*(stratified )?cv.*", "", s)
    s = re.sub(r",?\s*(held[- ]?out )?test set.*", "", s)
    s = re.sub(r",?\s*single held[- ]?out.*", "", s)
    s = re.sub(r"\s+", " ", s).strip(" ,")
    return s


df["metric_canonical"] = df["primary_metric_name"].apply(canonicalize)

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
        "modal_framing_share": round(top / n, 2),
    })
div_df = pd.DataFrame(diversity_rows).sort_values(["specificity", "hypothesis_id", "verify_arm"])
print(div_df.to_string(index=False))

print()
print("=" * 100)
print("SUMMARY: abstract vs concrete, collapsed across arms")
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

df.to_csv(EXPERIMENT_DIR / "results_with_canonical.csv", index=False)
print("\nSaved results_with_canonical.csv")
