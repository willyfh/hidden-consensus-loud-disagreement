"""Run a single isolated agent replicate against one hypothesis/arm.

Each replicate gets a fresh temp directory (no sibling replicate directories
visible in its parent), a static copy of the dataset, and a brief. It is
invoked via the headless Claude Code CLI with a fresh, memory-isolated
session. Results are collected into experiment/results/.
"""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hypotheses import build_brief

EXPERIMENT_DIR = Path(__file__).resolve().parent.parent
DATA_CSV = EXPERIMENT_DIR / "data" / "adult_income.csv"
RESULTS_DIR = EXPERIMENT_DIR / "results"
MODEL = "sonnet"
TIMEOUT_SECONDS = 900


def run_one(hyp_id: str, verify_arm: bool, replicate_idx: int, model: str = MODEL,
            results_dir: Path = RESULTS_DIR, timeout_seconds: int = TIMEOUT_SECONDS) -> dict:
    run_id = f"{model}_{hyp_id}_{'verify' if verify_arm else 'noverify'}_{replicate_idx:03d}_{uuid.uuid4().hex[:8]}"

    # Fresh, unpredictably-named temp dir with no sibling replicate dirs.
    work_dir = Path(tempfile.mkdtemp(prefix=f"aml_replicate_{run_id}_"))
    shutil.copy(DATA_CSV, work_dir / "adult_income.csv")

    brief = build_brief(hyp_id, verify_arm)

    meta = {
        "run_id": run_id,
        "hypothesis_id": hyp_id,
        "verify_arm": verify_arm,
        "replicate_idx": replicate_idx,
        "model": model,
        "work_dir": str(work_dir),
        "started_at": time.time(),
    }

    try:
        proc = subprocess.run(
            [
                "claude",
                "-p",
                brief,
                "--dangerously-skip-permissions",
                "--model",
                model,
            ],
            cwd=str(work_dir),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        meta["returncode"] = proc.returncode
        meta["stdout_tail"] = proc.stdout[-4000:]
        meta["stderr_tail"] = proc.stderr[-2000:]
        meta["timed_out"] = False
    except subprocess.TimeoutExpired as e:
        meta["returncode"] = None
        meta["stdout_tail"] = (e.stdout or "")[-4000:] if isinstance(e.stdout, str) else ""
        meta["stderr_tail"] = ""
        meta["timed_out"] = True

    meta["finished_at"] = time.time()
    meta["wall_clock_seconds"] = meta["finished_at"] - meta["started_at"]

    # Collect result.json + analysis.py if the agent produced them.
    result_json_path = work_dir / "result.json"
    analysis_py_path = work_dir / "analysis.py"

    meta["result_json_found"] = result_json_path.exists()
    meta["analysis_py_found"] = analysis_py_path.exists()

    run_result_dir = results_dir / run_id
    run_result_dir.mkdir(parents=True, exist_ok=True)

    if result_json_path.exists():
        try:
            parsed = json.loads(result_json_path.read_text())
            meta["parsed_result"] = parsed
            shutil.copy(result_json_path, run_result_dir / "result.json")
        except json.JSONDecodeError as e:
            meta["parse_error"] = str(e)
            shutil.copy(result_json_path, run_result_dir / "result_unparseable.json")

    if analysis_py_path.exists():
        shutil.copy(analysis_py_path, run_result_dir / "analysis.py")

    (run_result_dir / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    # Clean up the isolated work dir now that results are extracted.
    shutil.rmtree(work_dir, ignore_errors=True)

    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--replicate-idx", type=int, required=True)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result = run_one(args.hypothesis, args.verify, args.replicate_idx)
    print(json.dumps({k: v for k, v in result.items() if k not in ("stdout_tail", "stderr_tail")}, indent=2, default=str))
