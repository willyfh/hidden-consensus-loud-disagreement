"""Run the full replicate batch: 6 hypotheses x 2 arms x N replicates, in parallel.

Each replicate is fully independent (fresh temp dir, fresh CLI process, no
shared state) so parallelizing across threads is safe -- the only shared
resource is the local filesystem write to experiment/results/, which each
replicate writes to under its own unique run_id subdirectory.
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


def build_jobs():
    jobs = []
    for hyp_id in HYPOTHESES:
        for verify_arm in (False, True):
            for idx in range(N_REPLICATES):
                jobs.append((hyp_id, verify_arm, idx))
    return jobs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    total = len(jobs)
    print(f"Dispatching {total} replicate runs ({len(HYPOTHESES)} hypotheses x 2 arms x {N_REPLICATES}), "
          f"{MAX_WORKERS}-way parallel...", flush=True)

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
                }
            except Exception as e:
                failed += 1
                log_entry = {
                    "hypothesis_id": hyp_id,
                    "verify_arm": verify_arm,
                    "replicate_idx": idx,
                    "ok": False,
                    "exception": str(e),
                }
            done += 1
            log_f.write(json.dumps(log_entry, default=str) + "\n")
            log_f.flush()
            elapsed = time.time() - t0
            print(f"[{done}/{total}] {log_entry.get('run_id', f'{hyp_id}_{verify_arm}_{idx}')} "
                  f"ok={log_entry['ok']} elapsed={elapsed:.0f}s", flush=True)

    print(f"\nBatch complete: {done} runs, {failed} failed, {time.time()-t0:.0f}s total.", flush=True)


if __name__ == "__main__":
    main()
