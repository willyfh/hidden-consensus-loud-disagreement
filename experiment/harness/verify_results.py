"""Independently re-execute every replicate's saved analysis.py and check that the
reported primary_metric_value actually reproduces. This is a required disclosure step
(guarding against hallucinated results) -- separate from the verification *arm*, which is
an experimental condition, not our own check on the agents' numbers.

Pure local compute, no API calls -- safe to run against all successful replicates.
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = EXPERIMENT_DIR / "data" / "adult_income.csv"
RESULTS_DIR = EXPERIMENT_DIR / "results"
TIMEOUT_SECONDS = 900
MAX_WORKERS = 4  # each analysis.py may itself use n_jobs=-1 internally

# Each replicate's saved code may call RandomForestClassifier(n_jobs=-1) etc., which
# grabs every core via joblib. Combined with MAX_WORKERS-way process parallelism here,
# that oversubscribes the machine and can (confirmed empirically: two Sonnet replicates,
# 2026-08-17) produce a *different* value than a clean single-replicate run, despite a
# fixed random_state -- a false-positive "mismatch" caused by verification contention,
# not a real self-report/code discrepancy. Pin each subprocess to a small worker count
# so MAX_WORKERS-way outer parallelism can't oversubscribe the inner n_jobs=-1 calls.
_SUBPROCESS_ENV = os.environ.copy()
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "LOKY_MAX_CPU_COUNT"):
    _SUBPROCESS_ENV[_var] = "2"


def verify_one(run_dir: Path) -> dict:
    run_id = run_dir.name
    analysis_py = run_dir / "analysis.py"
    result_json = run_dir / "result.json"

    out = {"run_id": run_id, "has_analysis": analysis_py.exists(), "has_result": result_json.exists()}
    if not (analysis_py.exists() and result_json.exists()):
        out["status"] = "missing_files"
        return out

    try:
        original = json.loads(result_json.read_text())
    except json.JSONDecodeError:
        out["status"] = "original_unparseable"
        return out
    out["original_value"] = original.get("primary_metric_value")

    work_dir = Path(tempfile.mkdtemp(prefix=f"verify_{run_id}_"))
    shutil.copy(DATA_CSV, work_dir / "adult_income.csv")
    shutil.copy(analysis_py, work_dir / "analysis.py")

    try:
        proc = subprocess.run(
            ["python3", "analysis.py"],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            env=_SUBPROCESS_ENV,
            timeout=TIMEOUT_SECONDS,
        )
        out["returncode"] = proc.returncode
        out["stderr_tail"] = proc.stderr[-1500:]

        reexec_result_path = work_dir / "result.json"
        if reexec_result_path.exists():
            try:
                reexec = json.loads(reexec_result_path.read_text())
                out["reexecuted_value"] = reexec.get("primary_metric_value")
                out["found_in"] = "result.json"
                ov, rv = out["original_value"], out["reexecuted_value"]
                if isinstance(ov, (int, float)) and isinstance(rv, (int, float)):
                    out["abs_diff"] = abs(ov - rv)
                    out["match"] = abs(ov - rv) < max(0.01, 0.05 * abs(ov))
                else:
                    out["match"] = (ov == rv)
                out["status"] = "reexecuted"
            except json.JSONDecodeError:
                out["status"] = "reexecuted_output_unparseable"
        else:
            # Fallback: some replicates' analysis.py writes its detailed output under a
            # different filename (e.g. full_results.json) than the separately-authored
            # result.json. Search any JSON file the script produced for a numeric value
            # matching the original report before concluding it didn't reproduce.
            ov = out["original_value"]
            found = False
            if isinstance(ov, (int, float)):
                for jf in work_dir.glob("*.json"):
                    try:
                        data = json.loads(jf.read_text())
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        continue

                    def walk(node):
                        if isinstance(node, dict):
                            for v in node.values():
                                yield from walk(v)
                        elif isinstance(node, list):
                            for v in node:
                                yield from walk(v)
                        elif isinstance(node, (int, float)):
                            yield node

                    for val in walk(data):
                        if abs(val - ov) < max(0.001, 0.02 * abs(ov)):
                            out["reexecuted_value"] = val
                            out["found_in"] = jf.name
                            out["abs_diff"] = abs(val - ov)
                            out["match"] = True
                            found = True
                            break
                    if found:
                        break
            out["status"] = "found_in_other_file" if found else "no_result_json_produced"
    except subprocess.TimeoutExpired:
        out["status"] = "timeout"
    except Exception as e:
        out["status"] = "exception"
        out["exception"] = str(e)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    return out


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument("--out", default=str(Path(__file__).parent / "verification_log.jsonl"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
    print(f"Verifying {len(run_dirs)} result directories in {results_dir}, "
          f"{MAX_WORKERS}-way parallel...")

    out_path = Path(args.out)
    t0 = time.time()
    done = 0
    with out_path.open("w") as log_f, ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(verify_one, d): d for d in run_dirs}
        for fut in as_completed(futures):
            res = fut.result()
            log_f.write(json.dumps(res, default=str) + "\n")
            log_f.flush()
            done += 1
            print(f"[{done}/{len(run_dirs)}] {res['run_id']} status={res.get('status')} "
                  f"match={res.get('match')} elapsed={time.time()-t0:.0f}s")

    print(f"\nVerification complete: {done} checked, {time.time()-t0:.0f}s total.")


if __name__ == "__main__":
    main()
