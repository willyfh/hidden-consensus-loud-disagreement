"""Cross-model-family supplementary check: does the H1 (abstract) vs H2 (concrete)
measure-choice diversity pattern, and H2's directional finding, replicate with Haiku
in place of Sonnet? Compares the Haiku no-verify cells against the primary Sonnet
no-verify cells for the same two hypotheses.
"""
import json
import glob
import sys
from collections import defaultdict, Counter
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from canonicalize import canonicalize as _canonicalize


def canonicalize(hyp: str, name: str, mc=None) -> str:
    return _canonicalize(hyp, name, mc)


def load_haiku():
    d = defaultdict(list)
    for meta_path in glob.glob(str(EXPERIMENT_DIR / "results_haiku" / "*" / "meta.json")):
        m = json.load(open(meta_path))
        if not m.get("result_json_found"):
            continue
        key = (m["hypothesis_id"], m["replicate_idx"])
        d[key].append((m["started_at"], m))
    rows = []
    for key, entries in d.items():
        entries.sort()
        m = entries[0][1]
        parsed = m.get("parsed_result")
        if parsed:
            rows.append({
                "hypothesis_id": m["hypothesis_id"], "replicate_idx": m["replicate_idx"],
                "primary_metric_name": parsed.get("primary_metric_name"),
                "primary_metric_value": parsed.get("primary_metric_value"),
                "direction": parsed.get("direction"),
                "methodological_choices": parsed.get("methodological_choices"),
            })
    return pd.DataFrame(rows)


haiku_df = load_haiku()
haiku_df["metric_canonical"] = haiku_df.apply(
    lambda r: canonicalize(r["hypothesis_id"], r["primary_metric_name"], r.get("methodological_choices")), axis=1)

sonnet_df = pd.read_csv(EXPERIMENT_DIR / "results_with_canonical.csv")
sonnet_noverify = sonnet_df[sonnet_df["verify_arm"] == False]

print("=" * 90)
print("DIVERSITY COMPARISON: Sonnet vs. Haiku, no-verify arm, H1 (abstract) / H2 (concrete)")
print("=" * 90)
for hyp in ["H1", "H2"]:
    s = sonnet_noverify[sonnet_noverify["hypothesis_id"] == hyp]
    h = haiku_df[haiku_df["hypothesis_id"] == hyp]
    print(f"\n{hyp}: Sonnet unique canonical framings = {s['metric_canonical'].nunique()}/10, "
          f"Haiku = {h['metric_canonical'].nunique()}/10")
    print(f"  Sonnet modal framing share = {s['metric_canonical'].value_counts().iloc[0]/len(s):.0%}, "
          f"Haiku = {h['metric_canonical'].value_counts().iloc[0]/len(h):.0%}")

print()
print("=" * 90)
print("H2 DIRECTIONAL COMPARISON (RF vs. LogReg)")
print("=" * 90)
for name, df in [("Sonnet", sonnet_noverify[sonnet_noverify["hypothesis_id"] == "H2"]),
                  ("Haiku", haiku_df[haiku_df["hypothesis_id"] == "H2"])]:
    vals = df["primary_metric_value"].dropna()
    pos = (vals > 0).sum()
    neg = (vals < 0).sum()
    print(f"{name}: n={len(vals)}, mean={vals.mean():.4f}, positive(RF>LogReg)={pos}, "
          f"negative(LogReg>RF)={neg}, majority_share={max(pos,neg)/len(vals):.0%}")

haiku_df.to_csv(EXPERIMENT_DIR / "results_haiku_combined.csv", index=False)
print("\nSaved results_haiku_combined.csv")
