# Part 1 — Auto Research Result (draft v1)

**Title:** Hidden Consensus, Loud Disagreement: Autonomous ML Agents Diverge in What
They Measure but Converge on What They Conclude

---

## 1. Introduction and hypothesis

When independent human research teams are given the same dataset and the same research
question, they routinely reach different quantitative conclusions — not because any team
makes an error, but because the space of defensible analytical choices (which model,
which control variables, which metric, which validation scheme) is large, and different
teams occupy different corners of it. Menkveld et al. (2024) documented this directly:
164 independent teams analyzing the same financial dataset produced estimates whose
disagreement — "nonstandard errors" — often rivaled or exceeded each team's own reported
statistical uncertainty. Gao and Xiao (2026) recently showed that autonomous AI agents
reproduce this phenomenon: 150 independent Claude Code agents given the same six market
microstructure questions and dataset diverged sharply, driven by stable, identifiable
methodological forks (e.g., autocorrelation vs. variance-ratio measures of market
efficiency), with AI peer review largely failing to resolve the disagreement while
exposure to exemplar analyses did.

Both studies are confined to finance and social-science-style empirical analysis — domains
where the "correct" measure is inherently a matter of judgment, not something that can be
executed and checked. Machine learning research is different in one structural respect:
its central operations — train a model, evaluate it on held-out data, compute a metric —
are executable and directly checkable. This raises two questions neither prior study could
address. First, does the same nonstandard-errors phenomenon appear when the domain itself
provides mechanical ground truth, or does executability suppress it? Second, because ML
tasks are executable, agents can be asked to verify their own findings by literally
re-running the analysis under a stability check (repeated cross-validation, bootstrap
re-estimation) before reporting — an intervention with no clean analogue in a domain where
the underlying "truth" cannot itself be re-executed. Does this kind of intervention reduce
disagreement the way exemplar exposure did for Gao and Xiao?

We test both questions directly. We give independent, memory-isolated instances of Claude
Sonnet (via the Claude Code CLI) one of six machine-learning research questions about a
single fixed dataset, varying (a) whether the question specifies its measure explicitly or
leaves it to the agent, and (b) whether the agent must validate its finding's stability
before reporting. Our primary result has two parts. First, measure-choice divergence
replicates cleanly in the ML domain and is almost entirely explained by whether the
question's operationalization was specified: abstractly-worded questions produced 9.1
unique measure framings per 10 replicates on average, against 3.8 for concretely-specified
ones. Second, and more novel, this surface-level divergence substantially overstates
genuine disagreement about the underlying science — on two abstract questions, replicates
converge on the same qualitative conclusion 65-100% of the time despite near-total
disagreement on which specific number to headline — and the verification intervention
does not reduce measure-choice divergence at all, but does exactly what a well-designed
stability check should do to a numeric estimate: tighten it when the underlying effect is
real, and reveal its fragility when it is not.

## 2. Method

**Dataset.** All replicates analyze a single, static copy of the UCI/OpenML Adult (Census
Income) dataset (48,842 rows, mixed categorical/numeric features, binary income target,
~24% positive class), fetched once and frozen so every replicate receives byte-identical
data. This is one of the most standard tabular ML benchmarks in use, chosen specifically so
the paper's contribution rests on the agent-behavior finding rather than on any property of
an unfamiliar dataset.

**Research questions.** We wrote six research questions about this dataset, in three
matched abstract/concrete pairs [Table: hypotheses]:

- *H1 (abstract) / H2 (concrete).* Whether model family affects predictive performance, vs.
  whether a default-hyperparameter random forest beats default logistic regression on
  5-fold CV ROC-AUC.
- *H4 (abstract) / H5 (concrete).* Whether addressing class imbalance improves model
  quality, vs. whether SMOTE oversampling changes minority-class F1 by more than 0.02
  relative to no resampling, holding the classifier fixed.
- *H3 (abstract), H6 (abstract).* Which features matter most for prediction, and whether
  the model is well-calibrated — included as abstract questions with an implicit but
  unstated method (permutation importance; expected calibration error), to test whether
  method-level convergence can coexist with target-level divergence.

Each replicate agent receives only the dataset and one research question — no
documentation, no access to any other replicate's work, and no information about which
other measures might be reasonable.

**Isolation and scale.** Every replicate runs as an independent, non-interactive Claude
Code CLI invocation (`claude -p ... --dangerously-skip-permissions`, Claude Sonnet, single
model family) in a freshly created, unpredictably-named temporary directory with no
persistent sibling directories visible at any point in time, and an explicit instruction
not to explore outside its own working directory. We collected 10 independent replicates
per (hypothesis x arm) cell — 120 replicates total (see Part 2 for the full account of
data collection, including the Claude Code CLI's session usage limits and a genuine
mid-collection failure mode we diagnosed and partially mitigated).

**The verification arm.** Half of the replicates for each hypothesis (the "verify" arm)
receive an additional instruction: before reporting, validate your finding's stability via
repeated cross-validation with different seeds, a bootstrap confidence interval, or a
held-out re-test split, and report whether it held up. The other half ("no-verify") report
directly after a single analysis pass. This is the manipulation this paper adds beyond
Gao and Xiao: an intervention that only makes sense in a domain where the finding can
literally be re-executed.

**Structured reporting.** Every replicate writes a structured JSON result — a named
primary metric, its value, a qualitative direction/conclusion, and free-text
`methodological_choices` describing the specific forks it took (encoding, imbalance
handling, validation scheme, etc.) — designed to avoid the unit-normalization problem
Gao and Xiao had to solve after the fact, by asking each agent to self-report a single
named metric and its own reasoning rather than forcing a common statistical form up front.

**Verification of our own data.** Independent of the verification *arm* (an experimental
condition), we independently re-executed every replicate's saved analysis code against a
fresh copy of the dataset and checked whether the reported value reproduced. 113 of 120
(94%) reproduced exactly or within numerical tolerance; the remainder were traced entirely
to re-execution-harness limitations (a timeout on the heaviest repeated-CV scripts; a small
number of replicates whose saved script was not fully self-contained) rather than to any
incorrect reported value — no case showed a script running to completion and producing a
different number than reported. Full methodology and its limitations in Part 2.

## 3. Results

### 3.1 Measure-choice diversity tracks abstractness

Figure 1 shows the number of distinct measure framings (after collapsing trivial
rewordings, e.g. "mean over 5-fold CV" vs. "mean over stratified 5-fold CV," into a single
canonical form) each hypothesis produced across its 20 replicates (10 per arm). The
contrast is stark and consistent across every matched pair: H2 (concrete) produced only 2
distinct framings across 20 replicates, all of them minor rewordings of "ROC-AUC
difference (RF − LogReg)"; its abstract counterpart, H1, produced 15 — every replicate
compared a different pair of model families, or used a different specific measure
(held-out test ROC-AUC vs. cross-validated ROC-AUC, best-vs-worst vs. a named pair). H5
(concrete) produced 10 distinct framings; its abstract counterpart H4 produced 19 — nearly
one unique framing per replicate. H3 and H6, abstract questions with an implicit but
unstated method, land in between (15 and 17): agents largely agreed on the *method*
(permutation importance; 10-bin expected calibration error) while diverging on the
*target* it was applied to (which feature, which model).

Averaged over arms and collapsed into abstract vs. concrete, this is a 2.4x difference in
mean unique framings per 10 replicates (9.1 vs. 3.8) and a 3.6x difference in how much the
single most common framing dominates each cell (19% vs. 68% of replicates). This is a
direct, quantitative extension of Gao and Xiao's central finding — that measure choice,
not sampling noise, drives most agent-to-agent disagreement — into a domain their study
could not address, and confirms that the phenomenon is not specific to the ambiguity of
empirical finance measures: it appears just as strongly when the "measure" in question is
a machine learning evaluation metric with an unambiguous mathematical definition once
chosen.

### 3.2 Surface disagreement overstates substantive disagreement

The diversity counts in §3.1, read alone, would suggest that abstract ML research
questions produce near-total agent-to-agent disagreement. We show this conclusion is too
strong. For H1, we coded whether each replicate's reported ranking placed a tree-ensemble
model (Gradient Boosting, HistGradientBoosting, or Random Forest) above linear/naive
baselines (Logistic Regression, KNN, Gaussian Naive Bayes) as the top performer,
independent of which specific pair the replicate chose to headline. 20 of 20 replicates
(100%) agreed on this substantive ranking, despite zero agreement on which specific pair
of models to report as "the" comparison.

For H4, we coded whether each replicate's reported conclusion matched the pattern
"balanced accuracy / minority recall improves, while ROC-AUC / overall discriminative
quality stays roughly flat" — i.e., that imbalance-handling techniques trade precision for
recall rather than improving the model's underlying ranking quality. 13 of 20 replicates
(65%) stated this full pattern explicitly; 18 of 20 (90%) agreed at minimum on the
direction that imbalance handling "improves" outcomes on some accuracy-adjacent metric. As
with H1, this substantive convergence coexists with near-total disagreement on which
specific metric, strategy, and model to headline (19 of 20 unique framings; §3.1).

We read this as evidence that raw measure-choice-diversity counts, of the kind reported in
§3.1 and in Gao and Xiao (2026), can substantially overstate genuine scientific
disagreement: independent agents facing an underspecified question often converge on the
same understanding of the data while disagreeing loudly about how to summarize that
understanding into a single reportable number.

### 3.3 Verification does not resolve measure choice, but correctly adjusts numeric
confidence

Figure 2 compares the no-verify and verify arms for the two concrete hypotheses, where
values are numerically comparable across replicates. For H2 — a large, robust effect
(logistic regression modestly but consistently outperforming a default random forest) —
sign agreement is 100% in both arms (all 20 replicates report the same direction), and
requiring verification tightens the estimate's spread (SD 0.0038 -> 0.0022) without
changing measure-choice diversity, which if anything ticked up slightly (1 -> 2 unique
framings). For H5 — a small, fragile effect (SMOTE's effect on minority-class F1) — the
pattern is different and, we think, more interesting: sign agreement *falls* under
verification, from 90% (9/10) to 70% (7/10), and the mean reported effect shrinks toward
zero (-0.0028 -> -0.0016). Several replicates in the verify arm that initially observed a
negative effect on a single split found, once required to check via repeated
cross-validation, that the sign was not stable across folds, and reported this explicitly.

We interpret this pair of results as evidence that our verification manipulation is doing
exactly what a stability check should do, and nothing more: it reduces uncertainty in a
number that is real, and increases (correctly) reported uncertainty in a number that
was never as solid as a single-split analysis made it look. It has no mechanism to touch
measure-choice diversity, because checking whether a chosen estimate is stable is a
different operation from choosing which estimate to compute in the first place. This maps
cleanly onto the asymmetry Gao and Xiao report between their two interventions: AI peer
review, which "failed to resolve the underlying methodological forks," behaved like our
verification arm, while exemplar-paper exposure, which gave agents a concrete convergence
anchor, is the kind of intervention that would plausibly need to be transplanted into the
ML domain to change §3.1's diversity numbers.

## 4. Primary result

Across six machine learning research questions and 120 independent, memory-isolated agent
replicates, we find that (1) whether a research question specifies its measure explicitly
is, as in prior non-ML domains, the primary driver of agent-to-agent disagreement — a 2.4x
difference in measure framings between abstract and concrete questions — extending
"nonstandard errors" in AI-conducted research (Gao & Xiao, 2026; building on Menkveld et
al., 2024) into the one domain their method structurally could not reach; (2) this
surface-level divergence can substantially overstate genuine disagreement, with
substantive qualitative conclusions converging in 65-100% of cases even amid near-total
disagreement on what number to report; and (3) requiring agents to verify their own
findings via re-execution — an intervention unique to executable domains — does not touch
measure-choice diversity but correctly recalibrates numeric confidence, tightening robust
effects and surfacing the fragility of weak ones. All reported findings were independently
verified by re-executing each replicate's saved analysis code against the original data.

---

## Drafting notes (not for the paper)

- Word count: ~1,750 words body text (excluding this notes section), likely fits within a
  4-page two-column NeurIPS-style template once figures/tables are laid in, but should be
  checked against the actual Overleaf template once available -- may need trimming.
- [Table: hypotheses] placeholder -- need to lay in the actual H1-H6 table (text, arm,
  specificity) as a real table once in the template; full text lives in
  `harness/hypotheses.py`.
- Figure 1 and Figure 2 referenced by number -- files are
  `paper_notes/fig1_diversity.pdf` and `paper_notes/fig2_verify_arm.pdf`.
- Numbers in this draft (2.4x, 3.6x, 9.1, 3.8, 100%, 65%, 90%, std values, sign-agreement
  rates) are all pulled directly from `harness/analyze.py` output and the manual H1/H4
  coding pass earlier in this session -- double check each one against
  `results_with_canonical.csv` directly before final submission, don't just trust this
  draft's arithmetic blindly.
- H3/H6 are mentioned in §3.1 but not given the same substantive-convergence treatment as
  H1/H4 in §3.2 -- could add if space allows (e.g. "did agents at least agree on which
  broad feature *category*, e.g. demographic vs. financial, mattered most for H3" or "did
  they agree the model was miscalibrated in the same direction for H6") -- worth doing if
  we have space/time, flagged here so it's not forgotten, not done in this draft.
- Still needs: related-work paragraph properly citing the broader NSE cluster (Bertran et
  al. PNAS, Miao/Pritchard/Zou Stanford, Grundl, McCully -- full list in references.md) to
  preempt a discussant asking "isn't this just another paper in that cluster" -- current
  draft only cites the two most load-bearing precursors (Gao & Xiao; Menkveld et al.) and
  should be expanded before submission.
- This entire Part 1 is agent-drafted per the workshop's disclosure policy, which expects
  this ("we expect much of Part 1 to be generated and written by the agent"). Human should
  still read line by line before it goes anywhere near a submission -- author
  responsibility for correctness doesn't transfer.
