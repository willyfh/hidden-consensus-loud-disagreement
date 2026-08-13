# Reference tracker — A/ML workshop paper

Working list of papers surfaced during topic scoping (Aug 2026). Organized by relevance
tier so we know what must be cited/differentiated-from vs. background context. Titles/IDs
pulled from web search — **verify each citation (author list, venue, exact arXiv ID) before
the paper cites it**; search snippets are not a substitute for reading the source.

## Tier 0 — Direct precursor (must cite prominently, must differentiate from)

- **Nonstandard Errors in AI Agents** — Ruijiang Gao, Steven Chong Xiao. arXiv:2603.16744
  (March 2026). https://arxiv.org/abs/2603.16744
  Deploys 150 independent Claude Code agents (100 Sonnet 4.6, 50 Opus 4.6) to test 6
  hypotheses on NYSE TAQ SPY market microstructure data (2015–2024). Finds large
  agent-to-agent dispersion ("nonstandard errors") driven by stable methodological forks;
  AI peer review barely reduces dispersion; exemplar-paper exposure cuts IQR 80-99% in
  converging families but can also increase dispersion by introducing new forks. Explicitly
  states generalization beyond finance/market-microstructure to other domains is an open
  question for future work — this is the gap our paper targets. Domain (finance) is
  out-of-scope for this workshop, which is part of why the ML-domain extension is
  legitimate rather than a reskin.
  Design details worth reusing: per-agent filesystem/container isolation (agents were
  observed peeking at each other's outputs without it), abstract-vs-concrete hypothesis
  wording as the key explanatory variable for dispersion, effect-size unit normalization
  pipeline, ~$3/agent/stage cost.

- **Menkveld, Dreber, Holzmeister, Huber, Johannesson, Kirchler, Razen, Weitzel, et al.
  (2024)** — "Nonstandard Errors." *The Journal of Finance*, 79(3), 2339–2390.
  https://doi.org/10.1111/jofi.13337. Confirmed via direct search (2026-08-06) — large
  multi-author collaborative study, 164 independent human research teams analyze the same
  financial dataset/hypotheses; large cross-team dispersion in estimates ("nonstandard
  errors"), smaller for more reproducible/higher-rated research; peer-review stages reduce
  it. This is the human-researcher-teams study Gao & Xiao model their AI-agent design on
  (their "#fincap" → AI analog). Citation confirmed, no longer a TODO.

## Tier 1 — Closely related, needs explicit differentiation in Part 1/2

- **Preregistration for Experiments with AI Agents** — arXiv:2606.11217 (June 2026).
  https://arxiv.org/html/2606.11217v1
  Proposes preregistration methodology for AI agent experiments, explicitly framed around
  researcher-degrees-of-freedom concerns (model selection, prompt wording, outcome
  decisions). Methodology/proposal paper, not an empirical variance-quantification study —
  differentiate on that basis.

- **An Auditable AI Agent Loop for Empirical Economics: A Case Study in Forecast
  Combination** — arXiv:2603.17381. https://arxiv.org/pdf/2603.17381
  AI coding agents doing empirical specification search widen hidden researcher degrees of
  freedom; proposes agent-loop architecture with post-search holdout evaluation. Economics
  domain (out of scope here), but the holdout/verification idea is close to our
  executable-verification intervention — read closely before finalizing our intervention design.

- **MLReplicate: Benchmarking Autonomous Research Systems for Machine Learning
  Reproducibility** — arXiv:2605.16616 (May 2026). https://arxiv.org/abs/2605.16616
  Benchmarks 6 AI-scientist-style systems (AI Scientist v1/v2, Agent Laboratory,
  CycleResearcher, AI Researcher, Tiny Scientist) reproducing ICML 2025 outstanding papers;
  45 generated manuscripts, dual automated+human review. This is about *system capability*
  to reproduce *existing* published results — different question from ours (variance from
  underspecification on a *new* task), but will come up in related-work discussion of
  autonomous-research reliability.

- **REPA: Reproducibility Evaluation via an Autonomous ...** — OpenReview (title truncated
  in search results, need full retrieval). Similar reproducibility-benchmark territory to
  MLReplicate — check for overlap/distinction before citing.

- **From Reproduction to Replication: Evaluating Research Agents with Progressive Code
  Masking** — ICLR 2026, OpenReview. https://openreview.net/pdf?id=qBcHWGBnIb

- **Read the Paper, Write the Code: Agentic Reproduction of Social-Science Results** —
  arXiv:2604.21965. https://arxiv.org/html/2604.21965v1
  Agentic reproduction of social-science findings from methods+data (no author code).

- **AI Coding Agents Can Reproduce Social Science Findings** — arXiv:2606.11447.
  https://arxiv.org/html/2606.11447v1

## Tier 2 — Broader context on agent variance/consistency (background, cite selectively)

- **How Consistent Are LLM Agents? Measuring Behavioral Reproducibility in Multi-Step
  Tool-Calling Pipelines** — arXiv:2605.28840. https://arxiv.org/html/2605.28840
  2.3–4.2 distinct action sequences per 10 runs on identical inputs; run-to-run behavioral
  variance as a black-box uncertainty signal.

- **When Agents Disagree With Themselves: Behavioral Consistency as an Uncertainty Signal
  for LLM Agents** — arXiv:2602.11619. https://arxiv.org/html/2602.11619v2

- **Towards a Science of AI Agent Reliability** — arXiv:2602.16666.
  https://arxiv.org/abs/2602.16666

- **Can AI agents conduct open-ended AI research? Early evidence from two case studies** —
  arXiv:2607.27191. https://arxiv.org/abs/2607.27191

- **Questionable practices in machine learning** — arXiv:2407.12220 (2024). General survey
  of questionable research practices in ML (human researchers, pre-agent). Useful for
  framing "researcher degrees of freedom" concept lineage.

- **Examining the Effect of Implementation Factors on Deep Learning Reproducibility** —
  arXiv:2312.06633 (2023). Pre-agent-era reproducibility work — useful historical anchor,
  not competing.

## Tier 3 — Adjacent agent-behavior literature (context only, from earlier candidate exploration — likely not cited unless directly relevant)

- The Confidence Dichotomy: Analyzing and Mitigating Miscalibration in Tool-Use Agents —
  arXiv:2601.07264 / ACL 2026 (2026.acl-long.520)
- Honest Lying: Understanding Memory Confabulation in Reflexive Agents — arXiv:2605.29463
- Self-Correction Bench: Uncovering and Addressing the Self-Correction Blind Spot in LLMs —
  arXiv:2507.02778
- Governance Decay: How Context Compaction Silently Erases Safety Constraints in
  Long-Horizon LLM Agents — arXiv:2606.22528
- Why Do Multi-Agent LLM Systems Fail? — Cemri et al., arXiv:2503.13657
- Counterfactual Graph for Multi-Agent LLM Calibration — arXiv:2605.30653
- OAgents: An Empirical Study of Building Effective Agents — arXiv:2506.15741
- AgentRxiv: Towards Collaborative Autonomous Research — arXiv:2503.18102
- FIRE-Bench: Evaluating AI Agents on the Rediscovery of Scientific Insights —
  arXiv:2602.02905

## TODO

- [ ] Verify every citation above against the primary source (author list, exact venue,
      correct arXiv ID) before drafting Part 1 — search snippets can be wrong.
- [ ] Track down full Menkveld et al. (2024) citation.
- [ ] One more narrow novelty check specifically on "execution/verification-based
      mitigation of agent-to-agent result dispersion" before locking the intervention design.
- [ ] Re-check for new preprints in this cluster right before submission (Aug 22 abstract /
      Aug 29 final) — this space is moving on a ~weekly cadence.
