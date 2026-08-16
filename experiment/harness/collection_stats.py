"""Batch-dispatch statistics from the raw collection logs (Part 2, Table 5): attempts,
success rate, timeouts, retry-slot counts, and mean wall-clock time per model, across all
retry waves (not just the final N=10/cell dataset).
"""
import json
from collections import Counter
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent


def summarize(path, label):
    rows = [json.loads(l) for l in open(path)]
    total = len(rows)
    ok = sum(1 for r in rows if r["ok"])
    timeouts = sum(1 for r in rows if r.get("timed_out"))
    times = [r["wall_clock_seconds"] for r in rows if r["ok"]]
    slot_counts = Counter((r["hypothesis_id"], r["verify_arm"], r["replicate_idx"]) for r in rows)
    retried = sum(1 for c in slot_counts.values() if c > 1)
    print(f"{label}: attempts={total} succeeded={ok} ({ok/total:.0%}) timed_out={timeouts} "
          f"slots_needing_retry={retried}/{len(slot_counts)} ({retried/len(slot_counts):.0%}) "
          f"mean_wall_clock={sum(times)/len(times):.0f}s")


summarize(EXPERIMENT_DIR / "harness/batch_log.jsonl", "Sonnet")
summarize(EXPERIMENT_DIR / "harness/haiku_batch_log.jsonl", "Haiku")
