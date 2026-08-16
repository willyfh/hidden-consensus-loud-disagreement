# Representative examples

Full replicate output for a sample of runs referenced in the paper (Appendix C shows a
condensed version of the first two examples below). Each replicate writes a structured
`result.json` with four fields: `summary` (free-text conclusion), `primary_metric_name`
and `primary_metric_value` (the one number the replicate chose to headline), and
`methodological_choices` (free-text explanation of preprocessing/model/split/metric
decisions). Full files are in `results/<run_id>/result.json` and
`results_haiku/<run_id>/result.json`, alongside each replicate's saved `analysis.py`.

## H1 (abstract): "Does model family meaningfully affect predictive performance?"

Three replicates, no-verify arm, Sonnet. Same dataset, same question, same instructions —
three different model pairs and evaluation protocols, none reporting the same quantity.

### `H1_noverify_000_2d3da5bf`

- **Summary:** "Model family has a modest but real effect on predictive performance:
  gradient-boosted trees and random forest both outperform logistic regression by roughly
  0.014 and 0.010 ROC-AUC respectively (5-fold CV), and gradient boosting is the best of
  the three (CV ROC-AUC=0.9206 vs logistic regression=0.9066). The gap is noticeably
  larger than the fold-to-fold noise (std ~0.0035), so it reflects a genuine, if not huge,
  difference between model families on this dataset."
- **Metric:** ROC-AUC difference (GradientBoosting − LogisticRegression), 5-fold CV on
  training split = **0.0140**
- **Methodological choices (excerpt):** target binarized as `class=='>50K'`; missing
  categorical values imputed as a `Missing` category rather than dropped; numeric features
  standardized, categoricals one-hot encoded; 80/20 stratified split plus 5-fold CV on the
  training split; compared three model families (linear, bagged trees, boosted trees) at
  "reasonable defaults/light tuning, not exhaustively searched."

### `H1_noverify_001_7042393b`

- **Summary:** "Model family does meaningfully affect predictive performance on this
  dataset: across 5-fold stratified CV, mean ROC-AUC ranged from 0.8956 (KNN) to 0.9283
  (HistGradientBoosting), a gap of 0.0327 that is statistically significant (paired
  t-test p=2.464e-06)."
- **Metric:** ROC-AUC gap (HistGradientBoosting − KNN), 5-fold CV mean = **0.0327**
- **Methodological choices (excerpt):** `'?'` treated as missing and imputed (median for
  numeric, most-frequent for categorical) rather than dropped, "to retain all 48842 rows";
  four model families compared (Logistic Regression, k-NN with k=25, Random Forest,
  HistGradientBoosting), "all with mostly default" settings.

### `H1_noverify_004_d00ef2b5`

- **Summary:** "Model family has a modest but real effect on predictive performance:
  held-out ROC-AUC ranged from 0.8944 (KNN) to 0.9295 (HistGradientBoosting), a spread of
  0.0351 ... the choice of model family matters but is not the dominant driver of
  performance on this dataset."
- **Metric:** Test ROC-AUC spread (best − worst model family) = **0.0351**
- **Methodological choices (excerpt):** missing categoricals kept as an explicit `Missing`
  category; single 80/20 stratified split held out for the final comparison, plus 5-fold
  CV on the training set "for a more stable estimate per model."

Note what's *shared* across all three (target binarization, treatment of missing values as
informative rather than dropped, stratified splitting) versus what *diverges* (which model
pair gets named as "the" comparison, held-out test vs. CV, which of six-plus trained model
families gets picked as best/worst). The shared parts are why substantive conclusions
still converge (Section 1.3 of the paper); the diverging parts are measure-choice
diversity.

## H2 (concrete): "Does default-hyperparameter random forest beat default logistic
regression on stratified 5-fold CV ROC-AUC?"

### `H2_noverify_000_517aa10c` (no-verify arm)

- **Summary:** "Random forest achieved a mean stratified 5-fold CV ROC-AUC of 0.9023
  versus 0.9067 for logistic regression on the Adult Income dataset (lower by 0.0043), so
  the random forest does not outperform logistic regression on this metric."
- **Metric:** ROC-AUC difference (RF − LogReg), mean over 5 folds = **−0.0043**

### `H2_verify_000_def22a27` (verify arm — same underlying comparison, asked to check
stability first)

- **Summary:** "No — logistic regression achieves slightly higher stratified 5-fold CV
  ROC-AUC than random forest on this dataset (mean AUC 0.9067 for LogReg vs 0.9023 for RF,
  RF − LogReg = −0.0043). **This was confirmed by repeated CV and a fresh held-out
  split**, so with default hyperparameters for both models, RF does not beat LogReg here."
- **Metric:** ROC-AUC difference (RF − LogReg), mean over stratified 5-fold CV = **−0.0043**
  (identical to the no-verify replicate above, now with an explicit stability check
  attached)

This pair illustrates Section 1.3's "verification changes numeric confidence, not measure
choice" finding directly: same metric, same value, but the verify-arm replicate explicitly
re-ran the check (repeated CV plus a fresh split) before reporting, rather than reporting
after a single pass.

### Why does H5's sign agreement *fall* under verification (90% -> 70%)?

The paper (Section 1.3) attributes this to within-replicate fold instability: a verify-arm
agent runs repeated CV, sees the sign flip across folds, and honestly reports the
instability rather than a single-pass point estimate. That is real, but it is not the whole
story -- the raw data shows a second, distinct mechanism worth recording here.

Pulling `primary_metric_value` and every `random_state=` mention out of all 20 H5
replicates' `methodological_choices` text:

```
no-verify: -0.0009, -0.0007, -0.0075, -0.0009, -0.0065, -0.0015, -0.0037, -0.0047, -0.0015, 0.0003
verify:    -0.0012,  0.0005, -0.0052, -0.0027, -0.0007,  0.0009, -0.0004, -0.0047, -0.0042, 0.0019
```

18 of 20 replicates, in both arms, use `random_state=42` -- the scikit-learn-tutorial
default -- rather than each picking their own seed. That rules out "different random
train/test splits" as the main source of disagreement: most replicates are computing on
the *same* split of the *same* data. The spread in values above is therefore coming
predominantly from differences in how each replicate implemented the rest of the
pipeline -- imputation strategy, categorical encoding, exact SMOTE parameters, how "no
resampling" was implemented as the baseline -- not from split-to-split sampling noise.
This is the same kind of invisible-in-the-measure-name methodological fork the paper
documents for the Sonnet/Haiku H2 reversal (Section 1.3: naive label-encoding vs.
one-hot), just occurring within a single model family's independent replicates instead of
across model families.

Put together with the paper's own explanation: no-verify agents mostly compute one
estimate on the shared default split, and coincidentally agree on sign most of the time
(90%) because they're standing on the same split, even though their underlying
pipelines already differ. Verify-arm agents average over multiple folds/seeds per the
verification instruction, which (a) can reveal fold-to-fold instability within one
agent's own check (the paper's explanation), and (b) also washes out the shared-split
coincidence, leaving each agent's number driven more by its own pipeline's small,
genuinely-differing effect -- which for a near-zero true effect is enough to flip sign
more often across agents (70%). Both mechanisms point the same direction and are not
mutually exclusive; disentangling exactly how much each contributes would need a
controlled ablation (e.g. rerunning every replicate's saved `analysis.py` with several
different fixed seeds) that was not performed for this paper.

## H3 (abstract, implicit method): "Which features are most important for predicting
income?"

Agents converge on *method* (permutation importance) far more than on H1/H4, but still
diverge on which feature tops the list.

### `H3_noverify_000_cfc390f1`

- **Summary:** "Across a Random Forest (test ROC-AUC=0.917) and a Logistic Regression (test
  ROC-AUC=0.906), 'capital-gain' is consistently the single most important predictor of
  income by permutation importance, followed by 'education-num', 'age', and
  'marital-status'."
- **Metric:** permutation importance (ROC-AUC drop) of top feature ('capital-gain') =
  **0.0347**

### `H3_noverify_001_bf4b3e2a`

- **Summary:** "'capital-gain' is the single most important predictor of income class
  (mean ROC-AUC drop of 0.0486 when permuted), followed by relationship, education-num,
  age... A logistic-regression cross-check's raw coefficient ranking is dominated by
  native-country/occupation, but that is an artifact of summing many sparse one-hot
  coefficients... rather than genuine importance."
- **Metric:** top feature permutation importance (mean ROC-AUC drop) = **0.0486**

### `H3_noverify_002_9d44cada`

- **Summary:** "A random forest classifier (test ROC-AUC=0.918)... identifies
  'marital-status' as the most important predictor by permutation importance (mean
  test-set AUC drop of 0.0484 when shuffled), followed by capital-gain and
  education-num."
- **Metric:** top feature permutation importance (mean AUC drop) = **0.0484**
- **Direction:** 'marital-status' most important (vs. 'capital-gain' in the two replicates
  above) — same method, different named top feature.

## H4 (abstract): "Does addressing class imbalance improve model quality?"

### `H4_noverify_000_929e48a8`

- **Summary:** "Addressing class imbalance... does not improve ranking quality (ROC-AUC/
  PR-AUC are essentially unchanged) but it does trade accuracy and precision for
  substantially higher minority-class (>50K) recall, shifting the decision threshold
  rather than the underlying model."
- **Metric:** macro-F1 difference (RF class\_weight='balanced' − RF baseline) = **0.0064**

### `H4_noverify_001_7f9a4c8f`

- **Summary:** "Addressing class imbalance... barely changes ranking-based quality
  (ROC-AUC/PR-AUC virtually unchanged) but substantially shifts the precision/recall
  trade-off at the default 0.5 threshold."
- **Metric:** mean ROC-AUC delta (imbalance-handling strategies − baseline, averaged
  across LogReg/RF and ClassWeight/SMOTE) = **−0.0017**

### `H4_noverify_002_3bcfea33`

- **Summary:** "Addressing class imbalance (class\_weight='balanced') raised Random Forest
  balanced accuracy from 0.7805 to 0.8312 (delta=+0.0506) and minority-class F1 from
  0.6885 to 0.6979, while ROC-AUC stayed nearly flat."
- **Metric:** balanced accuracy difference (RF class\_weight='balanced' − RF baseline) =
  **0.0506**

All three substantively agree (imbalance handling shifts threshold-based metrics, not
ranking quality) despite headlining three different metrics (macro-F1, mean ROC-AUC delta,
balanced accuracy) at three different magnitudes — exactly the "surface disagreement
overstates substantive disagreement" pattern from Section 1.3.

## H5 (concrete): "Does SMOTE oversampling change minority-class F1 by more than 0.02
versus no resampling, holding the classifier fixed?"

Ten replicates on this hypothesis (e.g. `H5_noverify_000_aac7e2f2`) converge tightly on the
same metric name (minority-class F1 delta, SMOTE vs. no resampling) with modest numeric
spread — see Figure 2 and Table 2 in the paper for the full distribution; full
per-replicate `result.json` files are in `results/H5_noverify_*/`.

## H6 (abstract, implicit method): "Is the model well-calibrated?"

Agents converge on method (Expected Calibration Error, ~10 bins) far more than on H1/H4,
mirroring H3.

### `H6_noverify_000_16ee3b4d`

- **Summary:** "The primary model (HistGradientBoostingClassifier) is well-calibrated:
  Expected Calibration Error (10 equal-width bins) is 0.0059 and Brier score is 0.0869...
  in fact marginally better calibrated than a logistic-regression baseline (ECE=0.0088)."
- **Metric:** Expected Calibration Error (10 equal-width bins, HistGradientBoosting) =
  **0.0059**

### `H6_noverify_001_db82db8a`

- **Summary:** "The Gradient Boosting classifier is well-calibrated overall: predicted
  probabilities differ from observed outcome frequencies by only 0.96 percentage points on
  average (ECE)... though it shows mild overconfidence in the 0.7-0.9 predicted-probability
  range."
- **Metric:** Expected Calibration Error (10-bin, raw Gradient Boosting model) = **0.0096**

### `H6_noverify_002_0262d89f`

- **Summary:** "Yes, the model is well-calibrated: a gradient-boosted classifier... has an
  Expected Calibration Error of 0.008 (0.8 percentage points) across 10 probability bins
  on held-out test data."
- **Metric:** Expected Calibration Error (10-bin, Gradient Boosting, test set) = **0.008**

All three agree on method (ECE, ~10 bins) and land within 0.001-0.004 of each other
numerically, and all three reach the same qualitative conclusion (well-calibrated) — the
tightest substantive convergence of any abstract hypothesis in the study, consistent with
having an implicit conventional method rather than a fully open operationalization.

## Reproducing these examples

```
cat results/H1_noverify_000_2d3da5bf/result.json
cat results/H1_noverify_000_2d3da5bf/analysis.py   # the exact code that produced it
```

Every replicate directory follows this same structure; see `README.md` for the full
directory layout and `harness/hypotheses.py` for the exact brief text each replicate
agent received.
