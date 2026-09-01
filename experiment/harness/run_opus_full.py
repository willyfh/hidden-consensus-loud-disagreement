"""Third-model cross-model-family replication: all 6 hypotheses x 2 arms with Opus,
matching the primary Sonnet/Haiku design exactly. Resumable -- only dispatches cells
that don't already have a successful run logged, so a partial batch (if the full N=10
per cell isn't feasible in the available time) can be safely stopped and resumed, and
whatever completes is still a valid, evenly-distributed subset across the full
hypothesis space rather than being biased toward a few hypotheses.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hypotheses import HYPOTHESES
from run_replicate import run_one, EXPERIMENT_DIR

MODEL = "opus"
N_REPLICATES = 10  # matches Sonnet/Haiku design; reduce here for a smaller pilot
MAX_WORKERS = 8
RESULTS_DIR = EXPERIMENT_DIR / "results_opus"
LOG_PATH = Path(__file__).parent / "opus_batch_log.jsonl"
# Sonnet/Haiku used 900s; a first wave of real Opus replicates showed successful runs up
# to 1146s and several hitting the 900s wall (2026-08-31), so Opus needs materially more
# time per replicate. Widened here rather than in run_replicate.py's shared default, since
# that default is what Sonnet/Haiku's already-collected, already-analyzed data used.
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
    # Interleaved by replicate index across all (hypothesis, arm) cells first, rather than
    # exhausting one hypothesis before starting the next -- so if the batch is stopped early
    # (e.g. running long against a usage cap), whatever completed is spread evenly across the
    # full design instead of being biased toward the first hypotheses in dict order.
    ok_cells = cells_with_success()
    jobs = []
    for idx in range(N_REPLICATES):
        for hyp_id in HYPOTHESES:
            for verify_arm in (False, True):
                if (hyp_id, verify_arm, idx) not in ok_cells:
                    jobs.append((hyp_id, verify_arm, idx))
    return jobs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    total = len(jobs)
    print(f"Dispatching {total} Opus replicates (of {len(HYPOTHESES)*2*N_REPLICATES} total "
          f"design cells), {MAX_WORKERS}-way parallel...", flush=True)
    if total == 0:
        print("Nothing to do -- all cells already have a successful run.")
        return

    done = 0
    t0 = time.time()
    with LOG_PATH.open("a") as log_f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_one, hyp_id, verify_arm, idx, MODEL, RESULTS_DIR, TIMEOUT_SECONDS): (hyp_id, verify_arm, idx)
            for hyp_id, verify_arm, idx in jobs
        }
        for fut in as_completed(futures):
            hyp_id, verify_arm, idx = futures[fut]
            try:
                meta = fut.result()
                ok = meta.get("result_json_found") and not meta.get("timed_out")
                log_entry = {
                    "hypothesis_id": hyp_id, "verify_arm": verify_arm, "replicate_idx": idx,
                    "run_id": meta.get("run_id"), "ok": ok,
                    "returncode": meta.get("returncode"), "timed_out": meta.get("timed_out"),
                    "wall_clock_seconds": meta.get("wall_clock_seconds"),
                }
                if not ok:
                    log_entry["stdout_tail"] = meta.get("stdout_tail", "")[-300:]
            except Exception as e:
                log_entry = {"hypothesis_id": hyp_id, "verify_arm": verify_arm,
                             "replicate_idx": idx, "ok": False, "exception": str(e)}
            done += 1
            log_f.write(json.dumps(log_entry, default=str) + "\n")
            log_f.flush()
            print(f"[{done}/{total}] {log_entry.get('run_id', f'{hyp_id}_{verify_arm}_{idx}')} "
                  f"ok={log_entry['ok']} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\nDone: {done} runs, {time.time()-t0:.0f}s total.", flush=True)


if __name__ == "__main__":
    main()
