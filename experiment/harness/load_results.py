"""Load all successful replicate results into a single tidy CSV for analysis."""
import json
import glob
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
N_KEEP = 10  # uniform N per cell -- keep only replicate_idx < N_KEEP, assigned pre-outcome

rows = []
for meta_path in glob.glob(str(EXPERIMENT_DIR / "results" / "*" / "meta.json")):
    meta = json.load(open(meta_path))
    if not meta.get("result_json_found"):
        continue
    if meta["replicate_idx"] >= N_KEEP:
        continue
    parsed = meta.get("parsed_result")
    if not parsed:
        continue
    row = {
        "run_id": meta["run_id"],
        "hypothesis_id": meta["hypothesis_id"],
        "verify_arm": meta["verify_arm"],
        "replicate_idx": meta["replicate_idx"],
        "wall_clock_seconds": meta.get("wall_clock_seconds"),
        "summary": parsed.get("summary"),
        "primary_metric_name": parsed.get("primary_metric_name"),
        "primary_metric_value": parsed.get("primary_metric_value"),
        "direction": parsed.get("direction"),
        "methodological_choices": parsed.get("methodological_choices"),
        "verification_method": parsed.get("verification_method"),
        "verification_result": parsed.get("verification_result"),
    }
    rows.append(row)

df = pd.DataFrame(rows)
out_path = EXPERIMENT_DIR / "results_combined.csv"
df.to_csv(out_path, index=False)
print(f"Loaded {len(df)} rows -> {out_path}")
print(df.groupby(["hypothesis_id", "verify_arm"]).size())
