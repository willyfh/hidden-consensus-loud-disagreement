# Key findings summary (for drafting Part 1)

Final analysis on the clean N=120 dataset (12 cells x 10 replicates), verified via
independent re-execution (113/120 = 94% reproduced exactly, zero value mismatches among
completed re-runs -- see part2_system_notes.md for the verification methodology and its
known limitations).

## Finding 1: Measure-choice diversity tracks abstractness, replicating and extending
Gao & Xiao (2026) into the ML domain

| specificity | mean unique canonical framings / 10 | mean modal-framing share |
|---|---|---|
| abstract (H1,H3,H4,H6) | 9.125 | 18.75% |
| concrete (H2,H5) | 3.75 | 67.5% |

Per-cell breakdown in `harness/analyze.py` output. H1 and H4 show near-total framing
diversity (10/10 and 10/10 unique canonical framings in several cells); H2 and H5 show
strong convergence, dominated by one canonical phrasing.

## Finding 2: Surface diversity can mask substantive agreement

- H1: 100% of replicates (20/20 across both arms) rank a tree-ensemble model
  (GradientBoosting/HistGradientBoosting/RandomForest) above linear/naive baselines as
  the top performer, despite zero agreement on which specific model pair to headline as
  "the" comparison.
- H4: 65% of replicates explicitly state the full pattern "balanced accuracy/minority
  recall improves, ROC-AUC stays ~flat" (a recall/precision trade-off, not a real gain in
  discriminative power); ~90% agree on the "improves" direction at all, even though the
  specific headline metric/strategy varies almost completely.
- Implication: naive measure-choice-diversity counts (Finding 1) would overstate genuine
  scientific disagreement if read alone -- worth stating explicitly as a nuance beyond
  Gao & Xiao's framing, which measured effect-size dispersion but not this
  surface-vs-substance distinction.

## Finding 3: The verification-arm intervention changes numeric stability, not measure
choice -- and does so in a *content-appropriate* way

- H2 (large, robust effect: LogReg > RF on default hyperparams): 100% sign agreement in
  both arms; verification tightens the estimate (std 0.0038 -> 0.0022) without changing
  the measure-choice diversity (which if anything ticked up slightly).
- H5 (small, fragile effect: SMOTE vs no-resampling on minority F1): sign agreement
  *drops* under verification (90% -> 70%), and the mean effect shrinks toward zero
  (-0.0028 -> -0.0016). Read as verification correctly surfacing genuine fragility that a
  single-split analysis falsely presented as near-consensus, not as verification "failing."
- Theoretical framing for Part 1 discussion: verification-for-stability (repeated CV /
  bootstrap) targets *estimation noise* in an already-chosen measure; it has no mechanism
  to address *specification* disagreement (which measure to compute in the first place).
  This maps cleanly onto Gao & Xiao's own contrast between their weak intervention (AI
  peer review, which "failed to resolve the underlying methodological forks") and their
  strong intervention (exemplar exposure, which provides a convergence anchor and did
  work) -- our verification arm is structurally closer to their weak intervention in what
  it can plausibly fix.

## Candidate primary result statement for Part 1's abstract/intro

"Independent AI agents investigating identical ML research questions on identical data
diverge sharply in *what they choose to report* when the question is left abstract
(9.1 vs 3.8 unique measure framings per 10 replicates, abstract vs. concrete), replicating
and extending recent findings of 'nonstandard errors' in AI-agent-conducted empirical
research (Gao & Xiao 2026) into the machine learning domain their study could not cover.
However, this surface-level divergence substantially overstates disagreement about the
underlying science: on two abstract questions, replicates converge on the same
qualitative conclusion in 65-100% of cases despite near-total disagreement on which
specific number to headline. A mandatory self-verification step (repeated
cross-validation / bootstrap re-estimation) does not reduce measure-choice diversity, but
does appropriately tighten estimates for robust effects while surfacing genuine fragility
in weak ones -- consistent with verification targeting estimation noise rather than
specification choice."

## Data/artifacts for figures

- `results_with_canonical.csv` -- full 120-row dataset with canonicalized metric names,
  ready for figure generation.
- `harness/analyze.py` -- diversity table generation.
- Need to still produce: (1) a diversity bar chart (abstract vs concrete, per hypothesis),
  (2) H2/H5 value distributions by arm (e.g. strip/box plots), (3) maybe a small table of
  example raw `methodological_choices` text illustrating the specific forks.
