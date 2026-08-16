"""Why archived (excluded) replicate attempts were not kept, by model (Part 2, Table 7),
read directly from each archived run's saved meta.json rather than reconstructed after
the fact.
"""
import json
from collections import Counter
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent


def summarize(archive_dir, label):
    c = Counter()
    total = 0
    for meta_path in sorted(Path(archive_dir).glob("*/meta.json")):
        m = json.load(open(meta_path))
        total += 1
        if m.get("timed_out"):
            c["timed_out"] += 1
        elif not m.get("result_json_found"):
            c["no_result_file"] += 1
        elif m.get("result_json_found") and m.get("returncode") == 0:
            c["succeeded_extra_beyond_N10"] += 1
        else:
            c["other_error"] += 1
    print(f"{label} (N={total}):")
    for reason, n in c.most_common():
        print(f"  {reason}: {n} ({n/total:.0%})")


summarize(EXPERIMENT_DIR / "results_archive", "Sonnet")
summarize(EXPERIMENT_DIR / "results_haiku_archive", "Haiku")
