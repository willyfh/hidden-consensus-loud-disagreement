"""Hypothesis definitions for the NSE-in-ML-agents experiment.

Mix of abstract (operationalization left to the agent) and concrete
(explicit measure given) wording, mirroring Gao & Xiao (2026)'s key
explanatory variable for dispersion.
"""

HYPOTHESES = {
    "H1": {
        "specificity": "abstract",
        "text": "Does the choice of model family meaningfully affect predictive "
        "performance on this dataset?",
    },
    "H2": {
        "specificity": "concrete",
        "text": "Does a random forest classifier (scikit-learn defaults, "
        "RandomForestClassifier()) achieve higher stratified 5-fold "
        "cross-validated ROC-AUC than logistic regression (scikit-learn "
        "defaults, LogisticRegression()) on this dataset?",
    },
    "H3": {
        "specificity": "abstract",
        "text": "Which features are most important for predicting income in this dataset?",
    },
    "H4": {
        "specificity": "abstract",
        "text": "Does addressing class imbalance improve model quality on this dataset?",
    },
    "H5": {
        "specificity": "concrete",
        "text": "Does applying SMOTE oversampling to the training data change the "
        "minority-class (>50K) F1 score by more than 0.02 compared to no "
        "resampling, holding the classifier fixed as a default-hyperparameter "
        "random forest (RandomForestClassifier())?",
    },
    "H6": {
        "specificity": "abstract",
        "text": "Is the model well-calibrated?",
    },
}

BRIEF_TEMPLATE = """You are an independent data scientist. You have been given a single \
dataset and a single research question. Investigate it yourself, from scratch, and report \
your finding. You do not have access to any other researcher's work on this question — \
work entirely independently.

## Dataset

`adult_income.csv` is in your current directory. It is the UCI/OpenML Adult (Census Income) \
dataset: {n_rows} rows, columns are demographic and employment features, and the target is \
`class` (`<=50K` or `>50K`). No further documentation is provided — inspect the data \
yourself to understand it.

## Research question ({hyp_id})

{hyp_text}

## Important: this is a single-shot, non-interactive session

You will NOT receive a follow-up turn. Do not launch any script, computation, or shell \
command in the background and then wait for it / say you'll "check back" or "continue once \
notified" — there is no later turn in which that notification could reach you, and the \
process will simply end with your work incomplete. Run all code synchronously in the \
foreground and wait for each command to finish before proceeding. If an analysis is slow, \
that is fine — just run it in the foreground and let it take as long as it needs; do not \
background it.

## What to do

1. Write and execute your own Python analysis code to investigate this question. Make your \
own methodological choices (e.g. preprocessing, encoding, train/test split, model \
hyperparameters, evaluation metric, importance method — whatever is not explicitly \
specified above is your judgment call to make as the researcher).
{verification_block}
{write_step_num}. When finished, write your findings to a file named `result.json` in your current \
directory, with exactly this structure:

```json
{{
  "hypothesis_id": "{hyp_id}",
  "summary": "1-3 sentence plain-English answer to the research question",
  "primary_metric_name": "name of the single number that best answers the question, e.g. 'ROC-AUC difference (RF - LogReg)' or 'top feature permutation importance'",
  "primary_metric_value": <the numeric value, or null if not applicable>,
  "direction": "short phrase stating the finding's direction/conclusion, e.g. 'RF > LogReg' or 'capital-gain most important' or 'model is miscalibrated'",
  "methodological_choices": "free text: the specific choices you made that another researcher might have made differently (model class/params, encoding, validation scheme, metric/importance method, imbalance handling, etc.)"{verification_json_fields}
}}
```

{code_step_num}. Also save your analysis code as `analysis.py` in your current directory.

Do not read, list, or explore any directory other than your current working directory. \
Work only with the data and instructions given here.
"""

VERIFICATION_BLOCK = """{verify_step_num}. Before finalizing your answer, you must validate the stability of your primary \
finding — e.g. via repeated cross-validation with different random seeds, or a \
bootstrap confidence interval, or a held-out re-test split you did not use for your \
initial analysis. Report whether your finding held up under this check.
"""

VERIFICATION_JSON_FIELDS = """,
  "verification_method": "what you did to check stability (e.g. '5x repeated 5-fold CV with different seeds')",
  "verification_result": "did the finding hold up? include the revised estimate/range if it changed"
"""


def build_brief(hyp_id: str, verify_arm: bool, n_rows: int = 48842) -> str:
    hyp = HYPOTHESES[hyp_id]
    verify_step_num = 2
    write_step_num = 3 if verify_arm else 2
    code_step_num = write_step_num + 1
    verification_block = (
        VERIFICATION_BLOCK.format(verify_step_num=verify_step_num) if verify_arm else ""
    )
    return BRIEF_TEMPLATE.format(
        n_rows=n_rows,
        hyp_id=hyp_id,
        hyp_text=hyp["text"],
        verification_block=verification_block,
        write_step_num=write_step_num,
        code_step_num=code_step_num,
        verification_json_fields=VERIFICATION_JSON_FIELDS if verify_arm else "",
    )
