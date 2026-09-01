"""Opus collection scoped to H1 only (H2's abstract counterpart in the concrete/abstract
pairing), to check whether Opus's convergence with Sonnet on H2 also holds when Opus has
to make its own measure choice rather than following a fully-specified question. Reuses
the same resumable log/results as run_opus_full.py.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_replicate import run_one, EXPERIMENT_DIR

MODEL = "opus"
HYP_ID = "H1"
N_REPLICATES = 10
MAX_WORKERS = 8
RESULTS_DIR = EXPERIMENT_DIR / "results_opus"
LOG_PATH = Path(__file__).parent / "opus_batch_log.jsonl"
TIMEOUT_SECONDS = 2400


def cells_with_success():
    ok_cells = set()
    if LOG_PATH.exists():
        with LOG_PATH.open() as f:
            for line in f:
                d = json.loads(line)
                if d.get("ok"):
                    ok_cells.add((d["hypothesis_id"], d.get("verify_arm", False), d["replicate_idx"]))
    return ok_cells


def build_jobs():
    ok_cells = cells_with_success()
    jobs = []
    for idx in range(N_REPLICATES):
        for verify_arm in (False, True):
            if (HYP_ID, verify_arm, idx) not in ok_cells:
                jobs.append((HYP_ID, verify_arm, idx))
    return jobs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    total = len(jobs)
    print(f"Dispatching {total} Opus H1 replicates (of {N_REPLICATES*2} total), "
          f"{MAX_WORKERS}-way parallel...", flush=True)
    if total == 0:
        print("Nothing to do -- H2 already complete.")
        return

    done = 0
    t0 = time.time()
    with LOG_PATH.open("a") as log_f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_one, HYP_ID, verify_arm, idx, MODEL, RESULTS_DIR, TIMEOUT_SECONDS): (verify_arm, idx)
            for _, verify_arm, idx in jobs
        }
        for fut in as_completed(futures):
            verify_arm, idx = futures[fut]
            try:
                meta = fut.result()
                ok = meta.get("result_json_found") and not meta.get("timed_out")
                log_entry = {
                    "hypothesis_id": HYP_ID, "verify_arm": verify_arm, "replicate_idx": idx,
                    "run_id": meta.get("run_id"), "ok": ok,
                    "returncode": meta.get("returncode"), "timed_out": meta.get("timed_out"),
                    "wall_clock_seconds": meta.get("wall_clock_seconds"),
                }
                if not ok:
                    log_entry["stdout_tail"] = meta.get("stdout_tail", "")[-300:]
            except Exception as e:
                log_entry = {"hypothesis_id": HYP_ID, "verify_arm": verify_arm,
                             "replicate_idx": idx, "ok": False, "exception": str(e)}
            done += 1
            log_f.write(json.dumps(log_entry, default=str) + "\n")
            log_f.flush()
            print(f"[{done}/{total}] {log_entry.get('run_id', f'H2_{verify_arm}_{idx}')} "
                  f"ok={log_entry['ok']} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\nDone: {done} runs, {time.time()-t0:.0f}s total.", flush=True)


if __name__ == "__main__":
    main()
