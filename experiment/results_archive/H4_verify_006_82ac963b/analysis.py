"""
H4: Does addressing class imbalance improve model quality on the Adult
Income dataset?

Approach
--------
The target `class` is imbalanced (~76% <=50K, ~24% >50K). We compare, for
two model families (Logistic Regression and Random Forest), three ways of
handling that imbalance:
  1. baseline      - no imbalance handling
  2. class_weight  - inverse-frequency class weighting (`class_weight="balanced"`)
  3. SMOTE         - synthetic minority oversampling applied to the training
                     fold only (never to test data, to avoid leakage)

We evaluate on a held-out test set using metrics that are sensitive to
minority-class performance (F1 and recall for the minority class ">50K",
balanced accuracy) as well as threshold-independent ranking metrics
(ROC-AUC, average precision / PR-AUC) that should be largely unaffected by
reweighting/resampling since they don't depend on the decision threshold.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

# Missing values appear as NaN in workclass / occupation / native-country
# (originally encoded as "?"). Treat as their own "Missing" category rather
# than dropping rows, since >7% of rows would be lost otherwise.
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("class")
num_cols = [c for c in df.columns if c not in cat_cols + ["class"]]

X = df[cat_cols + num_cols].copy()
y = (df["class"].str.strip() == ">50K").astype(int)  # 1 = minority/positive class

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

print("Train class balance:\n", y_train.value_counts(normalize=True))
print("Test class balance:\n", y_test.value_counts(normalize=True))

preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
                    ("ohe", OneHotEncoder(handle_unknown="ignore")),
                ]
            ),
            cat_cols,
        ),
        ("num", StandardScaler(), num_cols),
    ]
)


def make_pipeline(model, resample=None):
    steps = [("prep", preprocess)]
    if resample == "smote":
        steps.append(("smote", SMOTE(random_state=RANDOM_STATE)))
        return ImbPipeline(steps + [("clf", model)])
    return Pipeline(steps + [("clf", model)])


MODEL_SPECS = {
    "logreg": lambda balanced: LogisticRegression(
        max_iter=1000,
        class_weight="balanced" if balanced else None,
        random_state=RANDOM_STATE,
    ),
    "rf": lambda balanced: RandomForestClassifier(
        n_estimators=200,
        max_depth=16,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced" if balanced else None,
        random_state=RANDOM_STATE,
    ),
}

STRATEGIES = ["baseline", "class_weight", "smote"]


def build_pipeline(model_name, strategy):
    if strategy == "class_weight":
        model = MODEL_SPECS[model_name](balanced=True)
        return make_pipeline(model, resample=None)
    elif strategy == "smote":
        model = MODEL_SPECS[model_name](balanced=False)
        return make_pipeline(model, resample="smote")
    else:  # baseline
        model = MODEL_SPECS[model_name](balanced=False)
        return make_pipeline(model, resample=None)


def evaluate(pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = pipe.predict(X_te)
    return {
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "f1_minority": f1_score(y_te, pred),
        "recall_minority": recall_score(y_te, pred),
        "precision_minority": precision_score(y_te, pred),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
    }


# ---------------------------------------------------------------------------
# 2. Primary comparison: single 70/30 held-out split
# ---------------------------------------------------------------------------
results = {}
for model_name in MODEL_SPECS:
    for strategy in STRATEGIES:
        pipe = build_pipeline(model_name, strategy)
        metrics = evaluate(pipe, X_train, y_train, X_test, y_test)
        results[f"{model_name}__{strategy}"] = metrics
        print(model_name, strategy, metrics)

results_df = pd.DataFrame(results).T
results_df.index.name = "model__strategy"
print("\n=== Held-out test set results (single 70/30 split) ===")
print(results_df.round(4))

# Deltas: imbalance-handled minus baseline, per model, per metric
deltas = {}
for model_name in MODEL_SPECS:
    base = results[f"{model_name}__baseline"]
    for strategy in ["class_weight", "smote"]:
        strat_metrics = results[f"{model_name}__{strategy}"]
        deltas[f"{model_name}__{strategy}_minus_baseline"] = {
            k: strat_metrics[k] - base[k] for k in base
        }
deltas_df = pd.DataFrame(deltas).T
print("\n=== Deltas vs baseline (positive = improvement) ===")
print(deltas_df.round(4))

# ---------------------------------------------------------------------------
# 3. Stability check: repeated stratified k-fold CV with multiple seeds
#    on the RandomForest baseline vs class_weight comparison (the model
#    family that gave the larger effect), using F1 (minority class) and
#    balanced accuracy as the metrics of interest.
# ---------------------------------------------------------------------------
print("\n=== Stability check: 5x repeated 5-fold CV (RandomForest) ===")

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=123)

cv_scores = {"baseline": {"f1": [], "bal_acc": [], "roc_auc": []},
             "class_weight": {"f1": [], "bal_acc": [], "roc_auc": []}}

for strategy in ["baseline", "class_weight"]:
    fold_f1, fold_bacc, fold_auc = [], [], []
    for train_idx, val_idx in rskf.split(X, y):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        pipe = build_pipeline("rf", strategy)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_val)[:, 1]
        pred = pipe.predict(X_val)
        fold_f1.append(f1_score(y_val, pred))
        fold_bacc.append(balanced_accuracy_score(y_val, pred))
        fold_auc.append(roc_auc_score(y_val, proba))
    cv_scores[strategy]["f1"] = fold_f1
    cv_scores[strategy]["bal_acc"] = fold_bacc
    cv_scores[strategy]["roc_auc"] = fold_auc
    print(
        f"{strategy}: F1={np.mean(fold_f1):.4f}+/-{np.std(fold_f1):.4f}, "
        f"BalAcc={np.mean(fold_bacc):.4f}+/-{np.std(fold_bacc):.4f}, "
        f"ROC-AUC={np.mean(fold_auc):.4f}+/-{np.std(fold_auc):.4f}"
    )

f1_diffs = np.array(cv_scores["class_weight"]["f1"]) - np.array(cv_scores["baseline"]["f1"])
bacc_diffs = np.array(cv_scores["class_weight"]["bal_acc"]) - np.array(cv_scores["baseline"]["bal_acc"])
auc_diffs = np.array(cv_scores["class_weight"]["roc_auc"]) - np.array(cv_scores["baseline"]["roc_auc"])

print(f"\nMean F1 delta (class_weight - baseline) across {len(f1_diffs)} folds: "
      f"{f1_diffs.mean():.4f} (std {f1_diffs.std():.4f})")
print(f"Mean BalAcc delta: {bacc_diffs.mean():.4f} (std {bacc_diffs.std():.4f})")
print(f"Mean ROC-AUC delta: {auc_diffs.mean():.4f} (std {auc_diffs.std():.4f})")
print(f"Fraction of folds where BalAcc improved with class_weight: {(bacc_diffs > 0).mean():.2f}")
print(f"Fraction of folds where F1 improved with class_weight: {(f1_diffs > 0).mean():.2f}")

# 95% CI on the balanced accuracy delta via normal approx across folds
ci_low = bacc_diffs.mean() - 1.96 * bacc_diffs.std(ddof=1) / np.sqrt(len(bacc_diffs))
ci_high = bacc_diffs.mean() + 1.96 * bacc_diffs.std(ddof=1) / np.sqrt(len(bacc_diffs))
print(f"Approx 95% CI for BalAcc delta: [{ci_low:.4f}, {ci_high:.4f}]")

# ---------------------------------------------------------------------------
# 4. Save summary
# ---------------------------------------------------------------------------
summary_out = {
    "held_out_results": {k: {mk: float(mv) for mk, mv in v.items()} for k, v in results.items()},
    "held_out_deltas": {k: {mk: float(mv) for mk, mv in v.items()} for k, v in deltas.items()},
    "cv_bal_acc_delta_mean": float(bacc_diffs.mean()),
    "cv_bal_acc_delta_std": float(bacc_diffs.std()),
    "cv_bal_acc_delta_ci": [float(ci_low), float(ci_high)],
    "cv_f1_delta_mean": float(f1_diffs.mean()),
    "cv_roc_auc_delta_mean": float(auc_diffs.mean()),
}
with open("cv_summary.json", "w") as f:
    json.dump(summary_out, f, indent=2)

print("\nDone. Summary written to cv_summary.json")
