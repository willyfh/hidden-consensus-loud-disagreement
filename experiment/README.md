# Supplementary material: Hidden Consensus, Loud Disagreement

Code and data for the paper's experiment: 240 independent, memory-isolated agent
replicates (120 Claude Sonnet, 120 Claude Haiku) investigating six machine-learning
research questions on the UCI/OpenML Adult (Census Income) dataset, testing whether
question specificity drives measure-choice diversity (it does, but concentrated in
specific questions rather than uniformly) and whether a self-verification manipulation
changes it. See `qualitative_examples.md` for full representative replicate output (the
condensed version in the paper's Appendix C is drawn from this file).

## Directory structure

- `harness/` -- all experiment code.
  - `hypotheses.py` -- the six research questions and the brief template given to each
    replicate agent (both arms).
  - `fetch_data.py` -- fetches the dataset once from OpenML and freezes a static local
    copy (`data/adult_income.csv`).
  - `run_replicate.py` -- runs one isolated replicate (fresh temp directory, headless
    Claude Code CLI invocation, result collection); takes a `model` parameter so the same
    code drives both the Sonnet and Haiku batches.
  - `run_batch.py` / `retry_failed.py` -- parallel dispatch of the Sonnet batch, and
    resumable retry of only the cells missing a successful replicate.
  - `run_haiku_full.py` / `run_haiku_supplement.py` -- the equivalent dispatch/retry
    scripts for the Haiku cross-model check.
  - `verify_results.py` -- independently re-executes every replicate's saved analysis
    code against the original data and checks the reported value reproduces
    (`--results-dir`/`--out` flags select Sonnet vs. Haiku; reproduces Table 6).
  - `load_results.py` / `load_haiku_results.py` -- collect all successful replicates into
    `results_combined.csv` / `results_haiku_combined.csv`.
  - `analyze.py` / `analyze_cross_model.py` / `analyze_haiku_supplement.py` -- the
    measure-choice-diversity and dispersion analysis reported in Section 1.3 (Figure 1,
    Table 2, Table 4).
  - `collection_stats.py` -- batch-dispatch attempt/success/retry/timing statistics from
    the raw logs (reproduces Table 5).
  - `archive_stats.py` -- reasons archived replicate attempts were excluded, read from
    each attempt's saved metadata (reproduces Table 7).
  - `make_figures.py` -- generates Figures 1 and 2.
  - `batch_log.jsonl` / `haiku_batch_log.jsonl` -- one line per dispatched replicate
    (hypothesis, arm, replicate index, success/failure, timing) across every collection
    wave, including retries.
  - `verification_log.jsonl` / `verification_log_haiku.jsonl` -- one line per replicate's
    independent re-execution check.
- `data/adult_income.csv` -- the frozen dataset (UCI/OpenML Adult, CC BY 4.0; see
  `https://archive.ics.uci.edu/dataset/2/adult`).
- `results/` / `results_haiku/` -- the 120+120 replicates used in the paper (one
  directory per replicate: `result.json`, `analysis.py`, `meta.json`).
- `results_archive/` / `results_haiku_archive/` -- every excluded, failed, or duplicate
  replicate attempt, kept for audit rather than deleted (see `meta.json` in each for why
  it was excluded).
- `results_combined.csv` / `results_with_canonical.csv` / `results_haiku_combined.csv` --
  the replicates flattened into tidy tables, with and without canonicalized measure-name
  text.
- `qualitative_examples.md` -- full free-text replicate output (summary, methodological
  choices) for a representative sample, not just metric names/values.
- `paper_notes/` -- working notes from drafting the paper (decision log, section drafts,
  key-findings summary, literature-scoping log) plus the figure outputs `make_figures.py`
  writes to; kept for provenance. `decisions_log.md` is the chronological record Part 2's
  human-interventions disclosure is drawn from. `references.md` is the tiered log of
  candidate related-work papers found during topic scoping, with dated verification
  notes; it backs Part 2's disclosed claim that roughly fifteen candidate research
  directions were checked against current literature before this one was pursued.
- `requirements.txt` -- exact Python package versions used.

## Reproducing the analysis (no new agent runs needed)

```
pip install -r requirements.txt
python harness/load_results.py            # rebuilds results_combined.csv from results/
python harness/load_haiku_results.py      # rebuilds results_haiku_combined.csv from results_haiku/
python harness/analyze.py                 # reproduces the Section 1.3 diversity numbers (Figure 1, Table 2)
python harness/analyze_cross_model.py     # reproduces the Sonnet-vs-Haiku comparison (Table 2, Table 4)
python harness/make_figures.py            # writes Figures 1 and 2 to paper_notes/
python harness/collection_stats.py        # reproduces the batch-dispatch statistics (Table 5)
python harness/archive_stats.py           # reproduces the archive-exclusion breakdown (Table 7)
python harness/verify_results.py --results-dir results --out harness/verification_log.jsonl
python harness/verify_results.py --results-dir results_haiku --out harness/verification_log_haiku.jsonl
                                           # re-executes every replicate's saved analysis.py
                                           # and checks its reported number reproduces (Table 6)
```

## Reproducing data collection (requires agent access)

Requires the Claude Code CLI (`npm install -g @anthropic-ai/claude-code`) authenticated
against Claude Sonnet and Claude Haiku (or comparable models). `python harness/fetch_data.py`
regenerates `data/adult_income.csv` from OpenML; `python harness/run_batch.py` and
`python harness/run_haiku_full.py` dispatch fresh Sonnet/Haiku batches respectively.
Because model behavior can drift across versions, bit-exact
reproduction of individual replicate outputs is not expected, but the qualitative pattern
(concentrated, question-specific measure-choice diversity rather than a uniform
abstract-vs-concrete split; verification tightening robust effects and surfacing fragile
ones) should be checkable with any comparably capable agent.
