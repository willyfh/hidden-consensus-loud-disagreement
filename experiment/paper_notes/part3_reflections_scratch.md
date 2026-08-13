# Part 3 scratch: reflection-worthy observations

Running list, captured as things happen rather than reconstructed later. Not a draft of
Part 3 — raw material the human will shape into the actual 1-page reflection.

- **The session-usage-cap incident (2026-08-05).** A laptop-plus-subscription setup hit a
  hard external usage wall at roughly 56 heavy-agentic-runs into a 180-run batch. This is
  a concrete, measurable ceiling on what "autonomous research" currently means outside a
  funded lab: Gao & Xiao ran 150 agents for $1,558 against metered API billing with no
  such wall; a solo researcher on a consumer plan hits a hard stop and has to design
  around reset windows. Worth naming directly — the gap between "an agent can conduct
  research end-to-end" and "an agent can conduct research end-to-end on hardware/access a
  single person actually has" is itself a finding about the current state of the field,
  not just an engineering inconvenience.
- **Isolation is easy to assert, harder to guarantee without infrastructure most solo
  researchers don't have.** Gao & Xiao needed containers after observing real
  cross-contamination risk. We approximated isolation with fresh temp directories and a
  prompt instruction, which is weaker and should be named as weaker, not glossed over.
  Question for Part 3: what's the minimum viable isolation guarantee for this kind of
  multi-replicate agent study, and is prompt-level isolation ever good enough to trust for
  a published result?
- **Novelty-checking itself was the single most time-consuming and humbling part of this
  process.** Multiple genuinely well-thought-out candidate topics turned out to already be
  covered — sometimes by papers just weeks old, once by a benchmark with the exact name
  we'd have picked ("STALE"). Worth reflecting on what this implies for autonomous
  research generally: if a competent agent+human pair spends real effort and still
  collides repeatedly with very recent prior art, that says something about the current
  pace/density of the field, and possibly about whether "check novelty via web search"
  is even an adequate method anymore given how fast preprints appear.
- (add more here as they come up during analysis/writing — don't wait until the end)
