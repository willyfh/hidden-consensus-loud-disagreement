# Abstract draft (v2)

**Working title:** Hidden Consensus, Loud Disagreement: Autonomous ML Agents Diverge in
What They Measure but Converge on What They Conclude

Independent AI agents (Claude Sonnet, via the Claude Code CLI) given identical machine
learning research questions and an identical dataset (UCI Adult/Census Income) diverge
sharply in what they choose to report when a question's operationalization is left
unspecified — abstractly-worded questions produced 9.1 unique measure framings per 10
replicates on average across 120 independent, memory-isolated replicate analyses, versus
3.8 for concretely-specified questions, extending recent findings of "nonstandard errors"
among AI agents conducting empirical research (Gao & Xiao, 2026) into machine learning
research itself, the one domain outside their finance/social-science study's scope. We
show, however, that this surface-level divergence substantially overstates genuine
disagreement: on two abstract questions, replicates converge on the same qualitative
conclusion in 65-100% of cases despite near-total disagreement on which specific number to
headline — loud disagreement about what to report can coexist with hidden consensus about
what is true. We additionally test a domain-appropriate intervention unavailable in prior
(non-executable) research settings — requiring agents to verify their own finding via
repeated cross-validation or bootstrap re-estimation before reporting — and find it does
not reduce measure-choice diversity, but appropriately tightens numeric estimates for
robust effects while surfacing genuine fragility in weak ones (sign agreement on a small
effect drops from 90% to 70% once agents are required to check it). All reported findings
were independently verified via re-execution of each replicate's saved analysis code
against the original data (94% reproduced exactly or within numerical tolerance, zero value
mismatches).

---

## Notes on this draft

- Word count: ~230 words, should fit typical workshop abstract limits (verify exact limit
  once submission portal / template details are confirmed).
- States the qualifying result explicitly per eligibility rules ("the paper must state
  the qualifying result in its abstract").
- Deliberately does not oversell: "extends... into the one domain their method could not
  directly address" is accurate and defensible, not "first ever" or "novel" framing that
  would invite an easy rebuttal citing the NSE cluster.
- Title changed from an earlier draft that opened with "Nonstandard Errors in AI Agents"
  -- an exact match to the precursor paper's (Gao & Xiao 2026) title, which would have
  read as a reskin before anyone got to the abstract. New title leads with our most
  distinctive contribution (surface divergence can mask substantive convergence) instead
  of borrowing the precursor's branding. Human caught this during review -- good catch,
  keep an eye out for this pattern (unintentionally echoing a close precursor's exact
  phrasing) elsewhere in the paper too, not just the title.
- TODO before final abstract submission:
  - Confirm exact word/character limit from the Overleaf template / OpenReview portal.
  - Add author list / affiliations per portal requirements.
