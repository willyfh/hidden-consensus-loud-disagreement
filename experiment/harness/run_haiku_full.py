"""Full cross-model-family replication: all 6 hypotheses x 2 arms x N=10 with Haiku,
matching the primary Sonnet design exactly. Resumable -- only dispatches cells that
don't already have a successful run logged (so the existing 20 H1/H2 no-verify
replicates from the initial supplement are kept, not redone).
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hypotheses import HYPOTHESES
from run_replicate import run_one, EXPERIMENT_DIR

MODEL = "haiku"
N_REPLICATES = 10
MAX_WORKERS = 8
RESULTS_DIR = EXPERIMENT_DIR / "results_haiku"
LOG_PATH = Path(__file__).parent / "haiku_batch_log.jsonl"


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
    for hyp_id in HYPOTHESES:
        for verify_arm in (False, True):
            for idx in range(N_REPLICATES):
                if (hyp_id, verify_arm, idx) not in ok_cells:
                    jobs.append((hyp_id, verify_arm, idx))
    return jobs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    total = len(jobs)
    print(f"Dispatching {total} Haiku replicates (of {len(HYPOTHESES)*2*N_REPLICATES} total "
          f"design cells), {MAX_WORKERS}-way parallel...", flush=True)
    if total == 0:
        print("Nothing to do -- all cells already have a successful run.")
        return

    done = 0
    t0 = time.time()
    with LOG_PATH.open("a") as log_f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_one, hyp_id, verify_arm, idx, MODEL, RESULTS_DIR): (hyp_id, verify_arm, idx)
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
