"""Core analysis: measure-choice diversity, numeric dispersion, and the verification-arm
comparison, per hypothesis.

Canonicalization logic lives in canonicalize.py (shared with analyze_cross_model.py and
analyze_haiku_supplement.py so all three stay in sync) -- see that module's docstring for
the uniform operationalization-choice rule and why it changed on 2026-09-30.
"""
from pathlib import Path
import sys

import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonicalize import canonicalize

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent

pd.set_option("display.max_colwidth", 120)
pd.set_option("display.width", 160)

df = pd.read_csv(EXPERIMENT_DIR / "results_combined.csv")

SPECIFICITY = {
    "H1": "abstract", "H2": "concrete", "H3": "abstract",
    "H4": "abstract", "H5": "concrete", "H6": "abstract",
}
df["specificity"] = df["hypothesis_id"].map(SPECIFICITY)

df["metric_canonical"] = df.apply(
    lambda r: canonicalize(r["hypothesis_id"], r["primary_metric_name"], r.get("methodological_choices")),
    axis=1,
)

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
