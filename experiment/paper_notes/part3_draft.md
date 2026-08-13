# Part 3 — Reflections (draft v1, target: 1 page / ~600 words)

Three things from this process seem worth the field's attention beyond this paper's
specific result.

**Novelty-checking may not scale to how fast the field is moving.** Before settling on
this paper's direction, we checked roughly fifteen candidate research questions against
current literature and found close, often very recent, prior art on nearly all of them —
in one case, a benchmark with the exact name we would have chosen ourselves. This was not
a failure of effort; it was several rounds of genuine web search against a field that is
now producing directly-overlapping work on a timescale of weeks, not years. If a
competent human-agent pair spending real effort on due diligence still collides
repeatedly with unseen prior art, "we checked and didn't find anything" is weaker evidence
of novelty than it used to be, for autonomous-research submissions specifically, since the
same tools that make a research loop fast also make everyone else's loop fast. Venues like
this one may need to treat rapid, honest convergent discovery — several groups
independently reaching adjacent results within weeks — as an expected feature of the
genre rather than a sign that something went wrong.

**A laptop and a subscription are not yet a research lab, and the gap is measurable, not
just inconvenient.** Collecting this paper's 120 replicates required five separate retry
waves across roughly 36 hours, driven by a session usage limit whose practical ceiling
turned out to be far smaller and more variable than expected. The most directly comparable
prior study ran 150 agents against metered API billing with no such wall. This is a
concrete data point on a question this workshop's premise raises implicitly: autonomous
research is not equally available to everyone who has an agent — access to a research lab
increasingly means access to a *budget*, and the size of that budget shapes what questions
are practically askable, not just how quickly they get answered.

**Verification is not one thing, and conflating its forms is a real risk.** This paper
required two structurally different verification steps that could easily be mistaken for
each other: an experimental manipulation testing whether *agents checking their own
numerical stability* changes what they report, and a separate, independent check that the
*numbers themselves* were not fabricated. Our result shows the first kind of check has a
specific blind spot — it cannot resolve disagreement about what to measure, only how
precisely a chosen measure was estimated — and we suspect this generalizes: as
"self-verification" becomes a standard mitigation for autonomous-agent unreliability, it
will be tempting to treat one plausible-sounding check as covering all the ways a result
could be wrong. It does not. Reviewers and discussants evaluating autonomous-research
submissions may need a habit of asking, specifically, which kind of wrongness a paper's
verification step could and could not have caught — a question with a different answer
for hallucinated numbers, unstable estimates, and disagreement about what should have been
measured at all.

---

## Drafting notes

- ~530 words in the body above, room to grow toward the ~1-page budget once laid into the
  actual template (exact capacity depends on font/margins there).
- Deliberately did not include a reflection on the disclosure-pressure conversation earlier
  in this session (the request to omit/misattribute process details, and the decision not
  to) -- it's genuinely relevant "how should the field adapt" material (the everyday
  pressure toward a cleaner-looking narrative, and what resisting it costs), but it's
  also the most sensitive thing to write about since it involves narrating a real
  disagreement in this session. Flagging it here as an option rather than including it
  unilaterally -- human's call whether it belongs in the final version, and if so, in what
  words.
- Could trim to 2 reflections if space is tight; the verification-conflation point is
  probably the most distinctive to this paper specifically (ties directly to the Part 1
  result) and the strongest candidate to keep if only two fit.

---

**CORRECTION (2026-08-08):** the "36 hours" figure above conflated calendar time with active compute time. Actual active-replicate-running time was only ~3-4 hours; the ~33-hour calendar span was mostly idle time between retry waves (partly session-limit waiting, partly human availability). The submitted `.tex` in `submission/` has the corrected figures — treat this draft file as superseded on this point.
