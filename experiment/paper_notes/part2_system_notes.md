# Part 2 raw material: system design

Structured notes for the human-written Part 2 (agent, harness, tools, research loop,
human interventions, verification). This file collects the raw facts; Part 2 itself still
needs to be written in prose by the human per the disclosure policy — this is source
material, not a draft of the section.

## Agent / model

- Claude Sonnet (via `--model sonnet` alias), invoked through the standalone Claude Code
  CLI (`@anthropic-ai/claude-code`, v2.1.221), not the VSCode extension session directly.
- Each replicate is a fresh, non-interactive invocation: `claude -p "<brief>"
  --dangerously-skip-permissions --model sonnet`, run with `cwd` set to an isolated temp
  directory and no `--add-dir` grants beyond that directory.
- `--dangerously-skip-permissions` was necessary for unattended batch execution (no human
  available to click through permission prompts across 90-180 runs); this is a real
  reduction in safety guardrails relative to interactive use, worth naming explicitly in
  Part 2's limitations.

## Harness

- Code lives in `experiment/harness/`: `hypotheses.py` (brief template + the 6 research
  questions), `run_replicate.py` (single isolated replicate runner), `run_batch.py` /
  `retry_failed.py` (parallel dispatch, 8-way via ThreadPoolExecutor, resumable retry
  keyed on which (hypothesis, arm, replicate_idx) cells already have a successful run).
- Isolation: fresh `tempfile.mkdtemp()` directory per replicate, dataset CSV copied in,
  brief passed as the CLI prompt argument (not a file the agent could stumble on
  alongside other artifacts), temp dir deleted after result extraction.
- Structured output contract: every replicate must write `result.json` with fixed fields
  (hypothesis_id, summary, primary_metric_name, primary_metric_value, direction,
  methodological_choices, + verification_method/verification_result in the verify arm)
  and save its `analysis.py`. This was designed specifically to avoid the unit-mismatch
  problem Gao & Xiao had to build a whole normalization pipeline to fix — by asking each
  agent to self-report a single named primary metric and its own methodological choices
  in free text, rather than trying to force a common statistical unit up front.
- Timeout: 900s per replicate (vs. Gao & Xiao's much longer allowance for a much bigger
  dataset/task); two replicates hit this ceiling during the incident batch (see decisions
  log) and are logged as failures, not silently dropped.

## Research loop / human role

- Human made every scope decision (topic, hypotheses sign-off, dataset, N, execution
  mechanism, model family scope, response to the usage-limit incident) — see
  `decisions_log.md` for the full chronological record with rationale.
- Agent (this Claude Code session) designed the brief templates, hypothesis wording,
  isolation mechanism, wrote all harness code, diagnosed the mid-batch failure by reading
  captured stdout rather than guessing, and built the resumable-retry mechanism.

## Verification steps

- **Status: done.** `harness/verify_results.py` independently re-executes every
  replicate's saved `analysis.py` against a fresh copy of the dataset and checks whether
  the reported `primary_metric_value` reproduces (exact match or within 1-5% numerical
  tolerance to allow for minor floating-point/threading nondeterminism in sklearn).
  Result: **113/120 (94%) independently reproduced**, with **zero cases of an actual
  value mismatch** — every discrepancy was either (a) a timeout in the re-verification
  harness itself (1 case, on a compute-heavy repeated-CV script), or (b) the replicate's
  saved `analysis.py` not being a fully self-contained reproduction of its process — some
  agents computed final numbers via inline tool calls and authored `result.json`
  separately, rather than having one script that regenerates everything end to end (6
  cases). This is a real, disclosable limitation of the verification method: saving
  "analysis code" doesn't guarantee that code alone reconstructs 100% of the agent's
  process. Worth stating plainly in Part 2 rather than glossed over.
- The verification *arm* (agents validating their own finding via repeated CV/bootstrap)
  is a separate thing from this — that's an experimental condition being studied, not our
  own check on whether the agents' numbers are real. Don't conflate the two in Part 2.
- Process note: the first re-verification attempt on the harder cases was run without the
  Python venv active (shell state doesn't persist across separate tool calls), causing
  spurious `ModuleNotFoundError` failures for anything using `imblearn`/SMOTE — corrected
  by re-running with the venv properly activated. Good example of why it's worth
  double-checking a "failure" is real before writing it down as a finding.

## Known limitations to disclose

- Isolation is prompt-level + filesystem-layout, not OS-level sandboxing (no
  containers/Singularity set up for this laptop run) — contrast with Gao & Xiao's
  container isolation, which they note was necessary after observing agents peek at
  sibling outputs without it.
- Single model family (Sonnet) for this pass; no cross-model-family "empirical styles"
  comparison like Gao & Xiao's Sonnet-vs-Opus analysis.
- Replicate count (N=10/cell target) is well below Gao & Xiao's 150; sized down for
  laptop/subscription-quota feasibility, not by statistical design. Disclose as a scope
  limitation affecting confidence-interval width.
- Hit a session usage cap mid-experiment; recovered via resumable retry rather than losing
  data, but this is itself evidence of a real constraint on laptop-scale autonomous
  research (candidate Part 3 material).
