# Decisions & human interventions log

Chronological record of decisions and interventions for Part 2's required disclosure
(agent role, prompts, tools, research loop, human interventions, verification steps).
Backfilled from the working conversation on 2026-08-04/05; will be updated live from
here on as decisions happen, not reconstructed after the fact.

## Topic selection (2026-08-04)

- Initial brainstorm covered ~15 candidate research questions across three genres:
  LLM-agent-behavior studies, mature classical-ML mechanisms, and large-scale systematic
  audits. Each was checked against current literature via web search before being
  accepted or rejected — see `references.md` for the full source list.
- Human rejected the first "silent researcher degrees of freedom" framing once we
  discovered it substantially overlapped with Gao & Xiao (2026) "Nonstandard Errors in
  AI Agents" and a cluster of adjacent 2026 papers (Bertran et al. PNAS, Miao et al.
  Stanford, Grundl, McCully).
- Human decision: rather than abandon the space, extend the NSE/multi-analyst
  methodology to the ML domain specifically — the one domain none of those papers cover,
  and the only domain this workshop's eligibility rules permit ("other scientific
  domains... out of scope"). This is the structural differentiation the paper rests on.
- Two more candidate genres were explored and rejected after novelty checks (agent
  self-verification/staleness studies; large-scale classical-ML audits) before returning
  to the NSE/ML-domain direction as the most defensible option given the time budget.

## Execution design (2026-08-04/05)

- **Human decision:** use the Claude Code CLI in headless/scripted mode
  (`claude -p ... --dangerously-skip-permissions`) rather than a separate Anthropic API
  key, to avoid managing separate billing and use the existing subscription.
  Trade-off disclosed to the human at the time: replicate count would be constrained by
  the plan's usage limits rather than a dollar budget — this constraint materialized
  later (see Incident below).
- **Human decision:** single model family for the first pass (Claude Sonnet) rather than
  comparing two families (e.g. Sonnet vs. Opus), to keep the first pass tractable given
  time/quota constraints. Explicitly deferred, not abandoned.
- Dataset: UCI/OpenML Adult (Census Income), fetched once and frozen to a static local
  CSV so every replicate sees byte-identical data (avoids per-replicate network
  dependency/flakiness, and matches the read-only static data design in Gao & Xiao).
- Six hypotheses (H1-H6) drafted, split 3 abstract / 3 concrete by design, mirroring
  Gao & Xiao's key explanatory variable (specificity of the research question). Full text
  in `harness/hypotheses.py`.
- Two arms: no-verification (agent reports its finding directly) and verification-required
  (agent must validate stability via repeated CV / bootstrap CI / held-out re-test before
  reporting) — this is the new intervention this paper adds relative to prior NSE work,
  testing whether ML's built-in executability reduces dispersion the way exemplar exposure
  did in Gao & Xiao's finance setting.
- Isolation approach: each replicate runs in a freshly created, unpredictably-named temp
  directory (no persistent sibling directories visible in a common parent at any point in
  time), with an explicit brief instruction not to explore outside its working directory.
  **Disclosed limitation:** unlike Gao & Xiao's container/Singularity isolation, this is a
  prompt-level + filesystem-layout mitigation, not an OS-level sandbox guarantee, given no
  container tooling was set up for this laptop-scale run. To be stated plainly in Part 2.

## Pilot phase (2026-08-04, ~23:45 KST)

- 3 pilot replicates run (H2 no-verify, H2 verify, H6 no-verify) to validate the mechanism
  before committing to a full batch. All 3 succeeded; verify-arm mechanics confirmed
  working (H2 verify replicate computed a 95% CI on the paired RF-vs-LogReg difference).
  Wall-clock: 119-232s per replicate.

## Batch sizing (2026-08-04)

- **Human decision:** N=15 replicates per cell (6 hypotheses x 2 arms = 180 total runs),
  chosen over N=10 for more statistical power, after being shown a preview of the
  cost/wall-clock tradeoff.
- **Human question, unresolved at launch:** whether the Claude Code plan has a usage cap
  to watch for. Human elected to proceed and watch for errors rather than pre-verify a
  limit.

## Incident: session usage limit hit mid-batch (2026-08-05, ~00:24 KST)

- Full 180-run batch dispatched at 8-way parallelism. Progress was healthy through
  roughly the first ~56 runs (H1 and most of H2), then 139/180 runs failed in rapid
  succession (~2-4s each, too fast to be genuine analysis attempts), plus two hard
  timeouts at the 900s cap.
- **Root cause found by inspecting a failed run's captured stdout:** "You've hit your
  session limit · resets 1:50am (Asia/Seoul)." This was a real, externally-imposed
  constraint, not a harness bug.
- Human asked directly whether anything had crashed (prompted by their machine feeling
  slow) — separately diagnosed as unrelated: system memory was fine (our processes used
  ~2-3GB against 38GB total RAM); the slowdown was attributed to Chrome by the human
  after closing it.
- **Human decision, once the reset time had passed:** rather than re-run the full batch,
  built and ran a retry script that resumes only the specific (hypothesis, arm,
  replicate_idx) cells that never got a successful run, preserving the ~56 already-good
  replicates.
- **Human decision:** reduce the per-cell target from N=15 to N=10 given the observed
  practical ceiling of ~56 successful heavy-agentic-runs per session window, to reduce
  the number of reset-wait cycles needed. Explicitly discussed and rejected going lower
  (N=5) on statistical grounds: n=5 per arm gives too little power to detect a
  verification-arm effect, which is the paper's key novel comparison, and IQR/dispersion
  estimates are unstable at that sample size.
- This incident is itself flagged as Part 3 material: the practical friction of running
  agent-based research at any real scale on a single consumer subscription/laptop is a
  concrete, disclosable constraint on how "autonomous" laptop-scale autonomous research
  currently is.

## Writing decision: how to describe the batch-sizing incident in the paper (2026-08-05)

- Human decision: Part 1 states the final design plainly — "N=10 independent replicates
  per hypothesis/arm cell" — with no narration of the N=15 attempt. This is standard
  methods-section practice (papers don't narrate hyperparameter-search history) and is
  not a disclosure issue since it's simply stating the design that was actually used.
- Part 2 gets one factual sentence about the research-loop event itself, since Part 2's
  specific remit (unlike Part 1) is human interventions in the research loop: the harness
  hit the CLI's session usage limit mid-batch and resumed via the retry mechanism without
  discarding completed runs. Agreed phrasing:
  > "During batch execution, the harness encountered the CLI's session usage limit
  > partway through; since each replicate's result is logged independently, the harness
  > resumed by re-running only cells without a completed replicate, so already-completed
  > runs weren't discarded. The final dataset uses N=10 replicates per hypothesis/arm cell."
  This does not name the abandoned N=15 target — that number isn't load-bearing for the
  disclosure, the event and final N are what matter. Nothing in this phrasing is false.
- Explicitly rejected: stating or implying N=10 was the a priori design from the start.
  That would misstate the actual decision history and isn't just omission — keep this
  boundary in mind if the wording gets revised later.

## Incident: headless-mode backgrounding bug discovered (2026-08-05, ~22:00 KST)

- After the second session-limit hit, inspected the retry batch's non-timeout failures
  closely and found a second, distinct failure mode affecting roughly 15/49 (~30%) of
  that batch's failures: compute-heavy replicates (e.g. repeated CV with many
  model/config combinations) would spawn their analysis as a background shell task and
  then end their turn with messages like "I'll wait for the notification" or "I'll
  continue once the scheduled wakeup fires" — the same backgrounding pattern used in
  interactive Claude Code sessions (including this very session). Because `claude -p` is
  single-shot and non-interactive, there is no later turn for such a notification to
  reach, so the process exits with the work incomplete and no `result.json` written.
- Fix: added an explicit instruction to the brief template (`hypotheses.py`) stating this
  is a single-shot session with no follow-up turn, and directing the agent to run
  everything synchronously in the foreground regardless of how long it takes, rather than
  backgrounding it. Applied before the next retry wave.
- This is worth keeping for Part 2/3: it's a concrete, previously-undocumented-by-us
  failure mode specific to running an interactively-trained agent harness in headless
  batch mode, discovered empirically by reading failed replicates' own transcripts rather
  than assumed in advance.

## Batch collection complete (2026-08-06)

- All 12 cells (6 hypotheses x 2 arms) now have at least N=10 successful replicates.
  Final tally: 137 total successful runs. H1 and H2 have 12-15 per cell (benefited from
  the original N=15 attempt before the usage-limit incident); H3-H6 have 10-11 per cell.
  Collection required 5 retry waves total across roughly 2026-08-04 23:00 through
  2026-08-06 06:00 KST, interleaved with session-usage-limit resets. Full incident
  history above.

## Dataset finalized to uniform N=10 (2026-08-06)

- Human decision: trim the uneven per-cell counts (H1/H2 had extra successes carried over
  from the original N=15 attempt) down to a uniform N=10 per cell for a clean 12x10=120
  design, rather than reporting an uneven 137. Selection rule: keep replicate_idx < 10 for
  every cell -- a pre-specified rule (indices were assigned at dispatch time, before any
  outcome was known), not a post-hoc/outcome-based selection.
- While doing this, found 3 genuine duplicate successes: the same (hypothesis, arm,
  replicate_idx) succeeded twice under different run_ids -- (H2,verify,7), (H2,verify,8),
  (H3,verify,4). Root cause: when `TaskStop` was used earlier to kill the N=15 retry batch
  before relaunching at N=10, the already-spawned `claude` subprocesses for those slots
  likely kept running as orphans and completed later; a subsequent retry wave, not yet
  seeing that success logged, independently dispatched a fresh job for the same slot, and
  both eventually succeeded. Deduped by keeping the earlier-started run and archiving the
  later duplicate. Worth a line in Part 2 as a concrete lesson about background-task
  lifecycle management in a resumable-retry harness.
- All excluded/failed/duplicate run directories were archived to `results_archive/`, not
  deleted -- `results/` now contains exactly the 120 runs used in the paper, while nothing
  is actually lost. The process history in this log and in `batch_log.jsonl` is unaffected
  by this cleanup and remains the full record regardless of what's in `results/`.

## Open items still to decide/do

- Build and run the verification pipeline (re-execute each replicate's saved `analysis.py`
  independently and confirm the reported numbers reproduce) before treating any number as
  final — required by the disclosure policy, not yet done.
- Decide whether to attempt a second model family (e.g. Sonnet vs. Haiku) given quota
  constraints, or keep single-family and disclose that as a scope limitation.

## Haiku dataset verification finding (2026-08-12)

- Ran the same independent re-execution verification pipeline on the Haiku dataset that
  was already applied to Sonnet. Result: 95/120 (79%) reproduced vs. Sonnet's 113/120
  (94%), and critically, 4/120 Haiku replicates showed a script running to completion and
  producing a value that CONTRADICTED its own report -- zero such cases existed in Sonnet.
- Manually re-ran two of the four mismatching scripts twice each, independently. Both
  gave identical output across both runs, ruling out simple run-to-run non-determinism
  (e.g. unseeded randomness, parallel-execution jitter). The saved code is deterministic;
  it just doesn't compute what was self-reported. This means for these replicates, the
  agent's result.json was not actually derived from executing its own saved analysis.py.
- Updated throughout the paper (abstract, Part 1 Method, Part 2 Verification section,
  Limitations, Disclosure Statement, checklist items referencing the old Sonnet-only
  94%/zero-mismatch figures) to report combined accurate numbers: 208/240 (87%) overall,
  0 Sonnet mismatches, 4 Haiku mismatches with the deterministic-but-inconsistent-report
  cause disclosed explicitly. Framed in Part 2 as consistent with the paper's own thesis:
  verification aimed at one failure mode (fabrication) can surface a different one
  (self-report/execution inconsistency) it wasn't specifically designed to find.
- This was caught during a routine final-review pass, not something we were looking for --
  worth remembering that "let's also verify the second dataset the same way" was almost
  skipped as redundant busywork, and instead surfaced the single most important integrity
  finding added this session.
