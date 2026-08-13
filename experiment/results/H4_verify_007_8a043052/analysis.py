"""
H4: Does addressing class imbalance improve model quality on the Adult Income dataset?

Pipeline:
1. Load & clean data (impute missing categoricals with 'Missing', binary target).
2. Single 70/30 stratified train/test split as the primary evaluation.
3. Two classifiers (Logistic Regression, Random Forest) x five imbalance strategies:
   none (baseline), class_weight='balanced', random oversampling, SMOTE, random undersampling.
4. Evaluate each of the 10 combinations on the held-out test set with a battery of
   metrics: ROC-AUC, PR-AUC (average precision), F1/precision/recall on the minority
   ('>50K') class, and balanced accuracy.
5. Pick the best-performing base model on the untouched baseline, then compare that
   model's baseline vs. its best imbalance-handling strategy.
6. Stability check: 5x repeated stratified 5-fold CV (5 different seeds) comparing
   baseline vs. best strategy for the chosen model, reporting mean +/- std across the
   25 folds for each metric.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                              precision_score, recall_score, balanced_accuracy_score,
                              make_scorer)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.under_sampling import RandomUnderSampler

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

cat_cols = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]
num_cols = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss",
            "hours-per-week"]

for c in cat_cols:
    df[c] = df[c].fillna("Missing")

y = (df["class"].str.strip() == ">50K").astype(int)
X = df[cat_cols + num_cols]

print("Class balance overall:")
print(y.value_counts(normalize=True), "\n")

# ---------------------------------------------------------------------------
# 2. Primary train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

def make_model(kind, imbalance):
    if kind == "logreg":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight="balanced" if imbalance == "class_weight" else None,
            random_state=RANDOM_STATE,
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced" if imbalance == "class_weight" else None,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        raise ValueError(kind)

    steps = [("prep", preprocess)]
    if imbalance == "oversample":
        steps.append(("sampler", RandomOverSampler(random_state=RANDOM_STATE)))
    elif imbalance == "smote":
        steps.append(("sampler", SMOTE(random_state=RANDOM_STATE)))
    elif imbalance == "undersample":
        steps.append(("sampler", RandomUnderSampler(random_state=RANDOM_STATE)))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def evaluate(model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    proba = model.predict_proba(X_te)[:, 1]
    pred = model.predict(X_te)
    return {
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "f1_pos": f1_score(y_te, pred),
        "precision_pos": precision_score(y_te, pred),
        "recall_pos": recall_score(y_te, pred),
        "balanced_acc": balanced_accuracy_score(y_te, pred),
    }

kinds = ["logreg", "rf"]
strategies = ["none", "class_weight", "oversample", "smote", "undersample"]

results = []
for kind in kinds:
    for strat in strategies:
        model = make_model(kind, strat)
        metrics = evaluate(model, X_train, y_train, X_test, y_test)
        row = {"model": kind, "strategy": strat, **metrics}
        results.append(row)
        print(f"{kind:8s} {strat:12s} " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

res_df = pd.DataFrame(results)
print("\nFull results table:")
print(res_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. Pick best base model on the untouched baseline (strategy == 'none'),
#    ranked by ROC-AUC (overall ranking quality, imbalance-agnostic).
# ---------------------------------------------------------------------------
baseline_rows = res_df[res_df.strategy == "none"].set_index("model")
best_model_kind = baseline_rows["roc_auc"].idxmax()
print(f"\nBest base model by baseline ROC-AUC: {best_model_kind}")

# Among imbalance strategies for that model, pick the best by balanced accuracy
# (balanced accuracy is directly sensitive to imbalance handling, unlike ROC-AUC).
model_rows = res_df[res_df.model == best_model_kind].set_index("strategy")
best_strategy = model_rows.drop(index="none")["balanced_acc"].idxmax()
print(f"Best imbalance strategy for {best_model_kind} by balanced accuracy: {best_strategy}")

baseline_metrics = model_rows.loc["none"]
best_metrics = model_rows.loc[best_strategy]

print(f"\nBaseline ({best_model_kind}, no imbalance handling):\n{baseline_metrics}")
print(f"\nBest imbalance-handled ({best_model_kind}, {best_strategy}):\n{best_metrics}")

delta_balanced_acc = best_metrics["balanced_acc"] - baseline_metrics["balanced_acc"]
delta_roc_auc = best_metrics["roc_auc"] - baseline_metrics["roc_auc"]
delta_f1 = best_metrics["f1_pos"] - baseline_metrics["f1_pos"]
delta_recall = best_metrics["recall_pos"] - baseline_metrics["recall_pos"]
delta_precision = best_metrics["precision_pos"] - baseline_metrics["precision_pos"]

print(f"\nDelta balanced_acc (best - baseline): {delta_balanced_acc:.4f}")
print(f"Delta roc_auc      (best - baseline): {delta_roc_auc:.4f}")
print(f"Delta f1_pos        (best - baseline): {delta_f1:.4f}")
print(f"Delta recall_pos    (best - baseline): {delta_recall:.4f}")
print(f"Delta precision_pos (best - baseline): {delta_precision:.4f}")

# ---------------------------------------------------------------------------
# 4. Stability check: 5x repeated stratified 5-fold CV (5 different seeds),
#    comparing baseline vs. best strategy for the chosen model, across the
#    FULL dataset (X, y) rather than only the single train split, so the
#    resampler is refit inside every fold.
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STABILITY CHECK: 5x repeated stratified 5-fold CV")
print("=" * 70)

scoring = {
    "roc_auc": "roc_auc",
    "balanced_acc": make_scorer(balanced_accuracy_score),
    "f1_pos": make_scorer(f1_score),
    "recall_pos": make_scorer(recall_score),
    "precision_pos": make_scorer(precision_score),
}

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=123)

cv_results = {}
for label, strat in [("baseline", "none"), ("best_strategy", best_strategy)]:
    model = make_model(best_model_kind, strat)
    cv = cross_validate(model, X, y, cv=rskf, scoring=scoring, n_jobs=-1)
    cv_results[label] = cv
    print(f"\n{label} ({strat}) across 25 folds:")
    for m in scoring:
        vals = cv[f"test_{m}"]
        print(f"  {m}: mean={vals.mean():.4f} std={vals.std():.4f} "
              f"[min={vals.min():.4f}, max={vals.max():.4f}]")

# Paired comparison per fold (since same folds used for both due to same rskf object
# being re-iterated deterministically with fixed random_state, the splits are identical)
bal_acc_baseline = cv_results["baseline"]["test_balanced_acc"]
bal_acc_best = cv_results["best_strategy"]["test_balanced_acc"]
diffs = bal_acc_best - bal_acc_baseline
print(f"\nPaired per-fold delta in balanced_acc (best_strategy - baseline) across 25 folds:")
print(f"  mean={diffs.mean():.4f} std={diffs.std():.4f} "
      f"95% CI (normal approx)=[{diffs.mean()-1.96*diffs.std()/np.sqrt(len(diffs)):.4f}, "
      f"{diffs.mean()+1.96*diffs.std()/np.sqrt(len(diffs)):.4f}]")
print(f"  fraction of folds where best_strategy > baseline: {(diffs > 0).mean():.2f}")

roc_auc_baseline = cv_results["baseline"]["test_roc_auc"]
roc_auc_best = cv_results["best_strategy"]["test_roc_auc"]
roc_diffs = roc_auc_best - roc_auc_baseline
print(f"\nPaired per-fold delta in roc_auc (best_strategy - baseline) across 25 folds:")
print(f"  mean={roc_diffs.mean():.4f} std={roc_diffs.std():.4f}")

print("\nDone.")
