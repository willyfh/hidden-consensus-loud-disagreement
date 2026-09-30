# Hidden Consensus, Loud Disagreement: When the ML Researcher Is an AI Agent

When independent AI agents are asked to answer the same machine-learning research
question from the same data, do they agree on what they find? Across 240 independent,
memory-isolated Claude Sonnet and Claude Haiku replicates investigating six research
questions on the UCI/OpenML Adult (Census Income) dataset, this paper finds that
*measure-choice diversity* (how many distinct ways replicates choose to measure the same
answer) is concentrated in specific questions rather than a uniform property of abstract
wording, that replicates often converge on the same qualitative conclusion despite loud
surface disagreement on what to report, that self-verification tightens numeric confidence
without resolving which measure was chosen, and that a cross-model check surfaces the
mirror-image failure: two models agreeing on what to measure while landing on opposite
answers, traced to an invisible preprocessing choice.

## Repository structure

This repository holds the code and data for the paper, referenced from its Reproducibility
Statement -- not the paper source itself.

- [`experiment/`](experiment/) -- all experiment code, data, and results; see
  [`experiment/README.md`](experiment/README.md) for the full directory layout and how to
  reproduce every table and figure in the paper.

## Reproducing the paper's numbers

See [`experiment/README.md`](experiment/README.md) for exact commands. In short: every
table and figure in the paper is regenerated directly from the 240 saved replicate
directories under `experiment/results/` and `experiment/results_haiku/` by the scripts in
`experiment/harness/` -- no step is hand-edited.

## License

The dataset is the UCI/OpenML Adult (Census Income) dataset, CC BY 4.0. Code is provided
for reproducibility of the paper.
