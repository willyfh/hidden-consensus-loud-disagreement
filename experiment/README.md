# Supplementary material: Hidden Consensus, Loud Disagreement

Code and data for the paper's experiment: 120 independent, memory-isolated Claude Sonnet
agent replicates investigating six machine-learning research questions on the UCI/OpenML
Adult (Census Income) dataset, testing whether question specificity drives
measure-choice divergence and whether a self-verification manipulation changes it.

## Directory structure

- `harness/` -- all experiment code.
  - `hypotheses.py` -- the six research questions and the brief template given to each
    replicate agent (both arms).
  - `fetch_data.py` -- fetches the dataset once from OpenML and freezes a static local
    copy (`data/adult_income.csv`).
  - `run_replicate.py` -- runs one isolated replicate (fresh temp directory, headless
    Claude Code CLI invocation, result collection).
  - `run_batch.py` / `retry_failed.py` -- parallel dispatch of the full batch, and
    resumable retry of only the cells missing a successful replicate.
  - `verify_results.py` -- independently re-executes every replicate's saved analysis
    code against a fresh copy of the data and checks the reported value reproduces.
  - `load_results.py` -- collects all successful replicates into `results_combined.csv`.
  - `analyze.py` -- the measure-choice-diversity and dispersion analysis reported in
    Section 1.3 of the paper.
  - `make_figures.py` -- generates Figures 1 and 2.
  - `batch_log.jsonl` -- one line per dispatched replicate (hypothesis, arm, replicate
    index, success/failure, timing) across every collection wave, including retries.
  - `verification_log.jsonl` -- one line per replicate's independent re-execution check.
- `data/adult_income.csv` -- the frozen dataset (UCI/OpenML Adult, CC BY 4.0; see
  `https://archive.ics.uci.edu/dataset/2/adult`).
- `results/` -- the 120 replicates used in the paper (one directory per replicate:
  `result.json`, `analysis.py`, `meta.json`).
- `results_archive/` -- every excluded, failed, or duplicate replicate attempt, kept for
  audit rather than deleted (see `meta.json` in each for why it was excluded).
- `results_combined.csv` / `results_with_canonical.csv` -- the 120 replicates flattened
  into a tidy table, with and without canonicalized measure-name text.
- `requirements.txt` -- exact Python package versions used.

## Reproducing the analysis (no new agent runs needed)

```
pip install -r requirements.txt
python harness/load_results.py      # rebuilds results_combined.csv from results/
python harness/analyze.py           # reproduces the Section 1.3 diversity/dispersion numbers
python harness/make_figures.py      # reproduces Figures 1 and 2
python harness/verify_results.py    # re-executes every replicate's saved analysis.py
                                     # and checks its reported number reproduces
```

## Reproducing data collection (requires agent access)

Requires the Claude Code CLI (`npm install -g @anthropic-ai/claude-code`) authenticated
against Claude Sonnet or a comparable model. `python harness/fetch_data.py` regenerates
`data/adult_income.csv` from OpenML; `python harness/run_batch.py` dispatches a fresh
batch of replicates. Because model behavior can drift across versions, bit-exact
reproduction of individual replicate outputs is not expected, but the qualitative pattern
(higher measure-choice diversity for abstract questions; verification tightening robust
effects and surfacing fragile ones) should be checkable with any comparably capable agent.
