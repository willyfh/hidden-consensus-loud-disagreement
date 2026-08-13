"""Small supplementary cross-model-family check: does the measure-choice diversity
pattern (H1 abstract vs. H2 concrete) generalize from Sonnet to Haiku?

N=10 replicates each, no-verify arm only, matching the Sonnet no-verify cells for H1/H2
exactly so the comparison is apples-to-apples. Separate results directory and run_id
prefix ("haiku_...") so this never mixes with the primary 120-replicate Sonnet dataset.
"""
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from run_replicate import run_one, EXPERIMENT_DIR

MODEL = "haiku"
HYPOTHESES = ["H1", "H2"]
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
                    ok_cells.add((d["hypothesis_id"], d["replicate_idx"]))
    return ok_cells


def build_jobs():
    ok_cells = cells_with_success()
    jobs = []
    for hyp_id in HYPOTHESES:
        for idx in range(N_REPLICATES):
            if (hyp_id, idx) not in ok_cells:
                jobs.append((hyp_id, idx))
    return jobs


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs()
    total = len(jobs)
    print(f"Dispatching {total} Haiku replicates ({len(HYPOTHESES)} hypotheses x "
          f"{N_REPLICATES}, no-verify only), {MAX_WORKERS}-way parallel...", flush=True)
    if total == 0:
        print("Nothing to do -- all cells already have a successful run.")
        return

    done = 0
    t0 = time.time()
    with LOG_PATH.open("a") as log_f, ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {
            pool.submit(run_one, hyp_id, False, idx, MODEL, RESULTS_DIR): (hyp_id, idx)
            for hyp_id, idx in jobs
        }
        for fut in as_completed(futures):
            hyp_id, idx = futures[fut]
            try:
                meta = fut.result()
                ok = meta.get("result_json_found") and not meta.get("timed_out")
                log_entry = {
                    "hypothesis_id": hyp_id, "verify_arm": False, "replicate_idx": idx,
                    "run_id": meta.get("run_id"), "ok": ok,
                    "returncode": meta.get("returncode"), "timed_out": meta.get("timed_out"),
                    "wall_clock_seconds": meta.get("wall_clock_seconds"),
                }
                if not ok:
                    log_entry["stdout_tail"] = meta.get("stdout_tail", "")[-300:]
            except Exception as e:
                log_entry = {"hypothesis_id": hyp_id, "replicate_idx": idx, "ok": False,
                             "exception": str(e)}
            done += 1
            log_f.write(json.dumps(log_entry, default=str) + "\n")
            log_f.flush()
            print(f"[{done}/{total}] {log_entry.get('run_id', f'{hyp_id}_{idx}')} "
                  f"ok={log_entry['ok']} elapsed={time.time()-t0:.0f}s", flush=True)

    print(f"\nDone: {done} runs, {time.time()-t0:.0f}s total.", flush=True)


if __name__ == "__main__":
    main()
