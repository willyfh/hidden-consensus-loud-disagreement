# Part 2 — System Design (draft v1)

*Drafting note: written by the agent from the session's process log
(`decisions_log.md`, `part2_system_notes.md`), for the human author to review, correct,
and take ownership of. Per this workshop's disclosure policy, the submitted paper should
note this collaborative drafting process in an author note or acknowledgment — see the
suggested line at the end of this file.*

---

## 1. The agent and harness

All experimental replicates were produced by Claude Sonnet, invoked through the standalone
Claude Code CLI (`@anthropic-ai/claude-code`, v2.1.221) rather than through the Anthropic
API directly — a deliberate choice to use the author's existing subscription rather than
provision separate metered billing, made explicitly aware of the tradeoff that replicate
throughput would then be governed by the CLI's session usage limits rather than a dollar
budget (a tradeoff that materialized during data collection; see §3).

Each replicate is a single, non-interactive invocation —
`claude -p "<brief>" --dangerously-skip-permissions --model sonnet` — run with its working
directory set to a freshly created, unpredictably-named temporary directory containing
only a static copy of the dataset. `--dangerously-skip-permissions` was necessary for
unattended batch execution across well over a hundred replicate runs, where no human was
available to approve tool-use prompts interactively; this is a real reduction in
interactive safety guardrails, relative to normal Claude Code use, that we think is worth
naming plainly rather than glossing over.

Isolation between replicates was approximated at two levels: a fresh, non-shared
filesystem location per replicate (no persistent sibling directories visible in a common
parent at any point in time), and an explicit brief instruction not to read, list, or
explore any directory other than the replicate's own working directory. This is weaker
than the container/Singularity-based isolation used in the most directly comparable prior
work (Gao & Xiao, 2026), which reports having observed agents reading sibling outputs
without such isolation during their own trial runs. We did not build container-level
isolation for this laptop-scale study, and treat this as a disclosed limitation rather than
a solved problem: we cannot rule out that prompt-level isolation alone is imperfect in ways
container isolation would have prevented, though we have no direct evidence of
cross-contamination in our data.

Every replicate was asked to produce a structured `result.json` (a named primary metric,
its value, a qualitative direction, and free-text `methodological_choices` describing its
specific forks) plus its analysis code as a saved `analysis.py`. This format was designed
specifically to sidestep the unit-normalization problem Gao & Xiao had to solve after data
collection (agents reporting effect sizes in incompatible units, requiring a dedicated
conversion pipeline) by asking each agent to self-report one named metric and its own
reasoning, rather than forcing a common statistical form up front.

## 2. Tools

The agent's own toolset within each replicate session was the standard Claude Code
toolset (file read/write, bash execution) with no additional MCP servers or external
tools configured; agents wrote and executed their own Python analysis code using whatever
libraries they chose to import (predominantly scikit-learn, pandas, and, for several
replicates addressing class imbalance, imbalanced-learn/SMOTE). On the orchestration side,
the batch harness itself — dispatching replicates, managing isolation, collecting results,
and the later verification and analysis passes — was written in Python (`harness/` in the
project directory) using standard library and scientific-Python tooling
(`concurrent.futures`, pandas, scikit-learn, matplotlib), run locally with no cloud
infrastructure beyond the Claude Code CLI calls themselves.

## 3. Research loop and human interventions

The research loop had three phases: topic selection, data collection, and analysis. Human
involvement was substantial and continuous throughout, not a single approval step at the
start.

**Topic selection.** The path to this paper's specific research question involved an
extended, iterative novelty-checking process: roughly fifteen candidate research questions
across three different genres (studies of LLM agent behavior/reliability; classical ML
mechanisms; large-scale systematic ML audits) were proposed and checked against current
literature via web search, and the great majority were found to already have close prior
art, in several cases published within weeks of when we checked. This is itself a
disclosable finding about the current pace of the field (§ Part 3), and is documented in
full in `references.md`. The specific direction pursued here — extending the
"nonstandard errors" methodology of Gao & Xiao (2026) into the machine learning domain —
was arrived at only after the human author explicitly rejected an earlier framing once we
discovered it substantially overlapped with that paper and an adjacent cluster of very
recent (2026) work, and directed the agent to find a structurally, not just
incidentally, distinct angle. The final scope (ML-domain hypotheses; the verification-arm
intervention) was a joint design, developed through direct back-and-forth between agent
proposals and human critique, redirection, and final sign-off on the hypothesis set,
dataset, and replication design.

**Data collection.** The human author made every scope decision during collection:
execution mechanism (Claude Code CLI over a raw API key), model-family scope (single
family, Sonnet, deferred rather than abandoned, for a first pass), and replicate count.
During batch execution, the harness encountered the CLI's session usage limit partway
through; since each replicate's result is logged independently, the harness resumed by
re-running only cells without a completed replicate, so already-completed runs weren't
discarded. The final dataset uses N=10 replicates per hypothesis/arm cell — a figure the
human author also explicitly defended against a further reduction to N=5 that the agent
had not proposed but the human considered and overruled on statistical grounds, since the
verification-arm comparison, the paper's most novel manipulation, would have had too
little power at that sample size.

Collection surfaced two genuine operational problems, both diagnosed by inspecting failed
replicates' own transcripts rather than assumed in advance. First, the CLI's session usage
limit was hit repeatedly — not once, but across five separate retry waves spanning
roughly 36 hours — with the practical per-window ceiling turning out to be far smaller and
more variable (roughly 15-56 successful heavy-agentic runs) than either party had
anticipated going in. The harness was made resumable specifically in response to this: a
retry script identifies exactly which (hypothesis, arm, replicate-index) cells lack a
successful run and re-dispatches only those, so no completed work was ever discarded across
the five waves. Second, a subset of replicates — disproportionately the more
compute-heavy verification-arm ones — were found to spawn their analysis as a background
shell task and then end their turn waiting for a completion notification that could never
arrive in a single-shot, non-interactive session, mirroring the interactive backgrounding
pattern the agent itself uses in ordinary Claude Code sessions. An explicit instruction
against this was added to the brief template; it measurably reduced but did not eliminate
the failure mode, which we report as observed rather than as a fully solved problem.

A separate, more mundane incident is worth recording as a lesson in harness lifecycle
management: three replicate slots were found, during final data cleaning, to have
succeeded twice under different run identifiers. The most likely explanation is that a
background process (`TaskStop`) used to halt an earlier retry wave terminated the
orchestrating script without killing already-spawned CLI subprocesses, which continued
running and eventually completed after a later retry wave had already, not yet seeing that
success logged, independently dispatched a fresh replicate for the same slot. Duplicates
were resolved by keeping the earlier-started run of each pair and archiving the other,
rather than deleting either — a pre-specified, non-outcome-based rule, since dispatch
order was fixed before any result was known.

The dataset was ultimately trimmed to a uniform N=10 replicates per cell (120 total) by
restricting to `replicate_idx < 10` in every cell — a rule fixed at dispatch time, before
any outcomes existed, applied for presentational clarity (an even 12x10 design) rather than
for any outcome-dependent reason. All replicates excluded by this rule, along with every
failed attempt, were archived rather than deleted, and remain available for audit.

**Analysis.** The human author reviewed the resulting draft figures and text at each stage
and, notably, caught a specific framing problem the agent had not flagged on its own: an
early draft paper title reused the precursor paper's exact title
("Nonstandard Errors in AI Agents") as a substring, which would have read as a derivative
reskin to any reader before they reached the abstract. The title was changed to lead with
this paper's own distinctive contribution instead. The human author also explicitly
rejected two requests-adjacent proposals during drafting — to retroactively describe the
N=10 replicate count as though it had been the design from the outset, and, separately, to
have Part 2 itself presented as solely human-authored without disclosing the agent's
substantial role in drafting it — on the grounds that both would misrepresent the
research process this section exists to document; the current text (including this
sentence) reflects the resolution of that discussion, not its omission.

## 4. Verification steps

Two distinct verification mechanisms appear in this paper and should not be conflated.

The **verification arm** is an experimental manipulation under study: replicate agents in
this arm are instructed to validate their own finding's stability (via repeated
cross-validation, a bootstrap confidence interval, or a fresh held-out re-test) before
reporting. Its effects are a result reported in Part 1, not a claim about data quality.

Separately, and specific to this section's remit, we independently verified the
**correctness of the underlying dataset** by re-executing every replicate's saved
`analysis.py` against a fresh copy of the source data and comparing the reproduced primary
metric to the one originally reported — a check made possible, and cheap (no further API
usage), by the fact that ML analyses are directly executable, unlike the empirical-finance
analyses in the most comparable prior work. 113 of 120 replicates (94%) reproduced exactly
or within small numerical tolerance. Critically, zero replicates showed a script running
to completion and producing a value that contradicted its own report — every discrepancy
traced to a limitation of the re-verification method itself: a subset of the heaviest
verification-arm scripts exceeded the re-verification timeout, and a smaller subset of
replicates had saved an `analysis.py` that was not fully self-contained (some agents
computed final numbers via intermediate tool calls and authored `result.json` separately,
rather than in one script that reproduces everything end to end). We disclose this as a
real limit on what "we re-ran the code" can prove — it establishes that reported numbers
were not fabricated on the runs it could fully check, not that every replicate's process
was perfectly self-documenting.

One further process note, useful as a small methodological lesson: an initial attempt at
re-verifying the harder cases was run without the project's Python virtual environment
active (a consequence of shell state not persisting across separate tool invocations),
producing spurious `ModuleNotFoundError` failures on any script using SMOTE. This was
caught by inspecting the actual error rather than accepting the failure at face value, and
corrected by re-running with the environment properly active. We mention it because it is
a concrete instance of the more general discipline this paper argues for: treat an
automated "failure" as a hypothesis to check, not a fact to record.

## 5. Significance of the main result

*[Drafting note: this subsection is the paper's own interpretive commentary on why the
Part 1 result matters — the part of Part 2 we'd most encourage the human author to write
or substantially rework in their own words, rather than adopt this draft as-is. A starting
point follows.]*

The result in Part 1 matters for two different audiences. For researchers using AI agents
as empirical collaborators, it is a direct, actionable warning: an underspecified request
to an AI agent does not fail loudly, it fails quietly, by returning a confident,
well-argued answer to a question subtly different from the one intended — and the fix
demonstrated here (specifying the measure, not just the goal) is cheap and available today.
For the autonomous-research community this workshop represents, the more important result
may be the second one: that a verification step targeting numerical stability does not,
and structurally cannot, resolve disagreement about what to measure in the first place.
As autonomous agents are given more latitude in real research pipelines, "have the agent
check its own work" is a natural first line of defense — this result suggests that defense
has a specific, identifiable blind spot, and that closing it likely requires a different
kind of intervention (in the spirit of Gao & Xiao's exemplar-exposure result) rather than a
more thorough version of the same one.

## 6. Known limitations

- Isolation between replicates is prompt-level and filesystem-layout-based, not an
  OS-level sandbox guarantee (no containers were used).
- Single model family (Sonnet) for this study; cross-model-family comparisons like
  Gao & Xiao's Sonnet-vs-Opus "empirical style" analysis were deliberately deferred, not
  attempted.
- N=10 replicates per cell, well below the 150 used in the most comparable prior study;
  sized for laptop/subscription-quota feasibility rather than by power analysis, which
  widens confidence in the reported estimates less than a larger study would.
- Our re-verification of replicate outputs, while extensive, could not fully check every
  replicate's process end-to-end for the reasons given in §4.

---

## Suggested disclosure line for the paper

*(Place in an author contributions note, footnote, or wherever the venue's template
expects agent-disclosure language — check the final submission instructions once posted.)*

> Part 2 of this paper was drafted collaboratively with the Claude Code agent from the
> project's process log, then reviewed, corrected, and revised by the human author, who
> takes full responsibility for its content.

---

**CORRECTION (2026-08-08):** the "36 hours" figure above conflated calendar time with active compute time. Actual active-replicate-running time was only ~3-4 hours; the ~33-hour calendar span was mostly idle time between retry waves (partly session-limit waiting, partly human availability). The submitted `.tex` in `submission/` has the corrected figures — treat this draft file as superseded on this point.
