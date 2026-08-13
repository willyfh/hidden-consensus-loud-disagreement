"""Retry only the failed (hypothesis, arm, replicate_idx) cells from batch_log.jsonl.

Reads the log, finds which cells never got a successful (ok=True) run, and
re-dispatches exactly those -- so a mid-batch failure (e.g. hitting a usage
cap) doesn't waste the runs that already succeeded.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hypotheses import HYPOTHESES
from run_replicate import run_one, RESULTS_DIR

N_REPLICATES = 10
MAX_WORKERS = 8
LOG_PATH = Path(__file__).parent / "batch_log.jsonl"


def cells_with_success():
    """(hyp_id, verify_arm, replicate_idx) -> True if we already have an ok run."""
    ok_cells = set()
    if LOG_PATH.exists():
        with LOG_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                if d.get("ok"):
                    ok_cells.add((d["hypothesis_id"], d["verify_arm"], d["replicate_idx"]))
    return ok_cells


def build_missing_jobs():
    ok_cells = cells_with_success()
    jobs = []
    for hyp_id in HYPOTHESES:
        for verify_arm in (False, True):
            for idx in range(N_REPLICATES):
                if (hyp_id, verify_arm, idx) not in ok_cells:
                    jobs.append((hyp_id, verify_arm, idx))
    return jobs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_missing_jobs()
    total = len(jobs)
    print(f"Retrying {total} missing/failed cells, {MAX_WORKERS}-way parallel...", flush=True)
    if total == 0:
        print("Nothing to retry -- all cells already have a successful run.")
        return

    done = 0
    failed = 0
    t0 = time.time()

    with LOG_PATH.open("a") as log_f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_one, hyp_id, verify_arm, idx): (hyp_id, verify_arm, idx)
            for hyp_id, verify_arm, idx in jobs
        }
        for fut in as_completed(futures):
            hyp_id, verify_arm, idx = futures[fut]
            try:
                meta = fut.result()
                ok = meta.get("result_json_found") and not meta.get("timed_out")
                if not ok:
                    failed += 1
                log_entry = {
                    "hypothesis_id": hyp_id,
                    "verify_arm": verify_arm,
                    "replicate_idx": idx,
                    "run_id": meta.get("run_id"),
                    "ok": ok,
                    "returncode": meta.get("returncode"),
                    "timed_out": meta.get("timed_out"),
                    "wall_clock_seconds": meta.get("wall_clock_seconds"),
                    "retry": True,
                }
                if not ok:
                    log_entry["stdout_tail"] = meta.get("stdout_tail", "")[-300:]
            except Exception as e:
                failed += 1
                log_entry = {
                    "hypothesis_id": hyp_id,
                    "verify_arm": verify_arm,
                    "replicate_idx": idx,
                    "ok": False,
                    "exception": str(e),
                    "retry": True,
                }
            done += 1
            log_f.write(json.dumps(log_entry, default=str) + "\n")
            log_f.flush()
            elapsed = time.time() - t0
            print(f"[{done}/{total}] {log_entry.get('run_id', f'{hyp_id}_{verify_arm}_{idx}')} "
                  f"ok={log_entry['ok']} elapsed={elapsed:.0f}s", flush=True)

    print(f"\nRetry complete: {done} runs, {failed} failed, {time.time()-t0:.0f}s total.", flush=True)


if __name__ == "__main__":
    main()
