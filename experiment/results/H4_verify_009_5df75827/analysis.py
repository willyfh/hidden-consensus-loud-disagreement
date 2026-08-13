"""
H4: Does addressing class imbalance improve model quality on the Adult Income dataset?

Approach
--------
The target `class` is moderately imbalanced (~76% <=50K, ~24% >50K).
We compare a baseline classifier (no imbalance handling) against three
common imbalance-handling strategies:
  1. class_weight='balanced' (reweighting the loss)
  2. Random oversampling of the minority class (train fold only)
  3. SMOTE synthetic oversampling of the minority class (train fold only)

We do this for two model families (Logistic Regression, Random Forest) to
check whether any effect is model-specific. Because ROC-AUC is a ranking
metric that is largely insensitive to class-imbalance-handling techniques
(they mostly shift the decision threshold rather than the ranking), we
report ROC-AUC as well as threshold-sensitive metrics (F1 and recall for
the minority ">50K" class, and balanced accuracy) since those are where
imbalance-handling is expected to matter most in practice.

Primary metric: macro-F1 (balances precision/recall across both classes
and is sensitive to the minority-class behavior that rebalancing targets),
compared between the baseline and the best imbalance-handling strategy,
evaluated via 5-fold stratified cross-validation on a held-out test set
confirmation.

Validation of stability: 5x repeated stratified 5-fold CV (5 different
random seeds) on the full training data, comparing baseline vs class_weight
balanced Random Forest, reporting mean +/- std of the macro-F1 and ROC-AUC
difference.
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
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import RandomOverSampler, SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & inspect
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

TARGET = "class"
y = (df[TARGET] == ">50K").astype(int)
X = df.drop(columns=[TARGET])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = [c for c in X.columns if c not in cat_cols]

print("Rows:", len(df))
print("Class balance:", y.value_counts(normalize=True).to_dict())
print("Categorical cols:", cat_cols)
print("Numeric cols:", num_cols)

# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
#    - categorical: impute missing with a constant "Missing" category, one-hot encode
#    - numeric: median impute (none needed here but safe), standard scale
# ---------------------------------------------------------------------------
cat_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
num_pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
    ("scale", StandardScaler()),
])
preprocess = ColumnTransformer([
    ("cat", cat_pipe, cat_cols),
    ("num", num_pipe, num_cols),
])

# ---------------------------------------------------------------------------
# 3. Train/test split (held out for final confirmation only)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

def make_model(kind, imbalance_strategy):
    if kind == "logreg":
        clf = LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced" if imbalance_strategy == "class_weight" else None,
        )
    elif kind == "rf":
        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            class_weight="balanced" if imbalance_strategy == "class_weight" else None,
        )
    else:
        raise ValueError(kind)

    if imbalance_strategy == "smote":
        pipe = ImbPipeline([
            ("prep", preprocess),
            ("resample", SMOTE(random_state=RANDOM_STATE)),
            ("clf", clf),
        ])
    elif imbalance_strategy == "oversample":
        pipe = ImbPipeline([
            ("prep", preprocess),
            ("resample", RandomOverSampler(random_state=RANDOM_STATE)),
            ("clf", clf),
        ])
    else:
        # "none" (baseline) or "class_weight" (handled inside the estimator, no resampling)
        pipe = Pipeline([
            ("prep", preprocess),
            ("clf", clf),
        ])
    return pipe


scoring = {
    "roc_auc": "roc_auc",
    "f1_macro": "f1_macro",
    "f1_minority": "f1",
    "recall_minority": "recall",
    "balanced_acc": "balanced_accuracy",
}

# ---------------------------------------------------------------------------
# 4. Cross-validated comparison on the training set: baseline vs 3 strategies,
#    for both model families
# ---------------------------------------------------------------------------
strategies = ["none", "class_weight", "oversample", "smote"]
model_kinds = ["logreg", "rf"]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []
for kind in model_kinds:
    for strat in strategies:
        pipe = make_model(kind, strat)
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": kind, "imbalance_strategy": strat}
        for metric in scoring:
            row[f"{metric}_mean"] = float(np.mean(cvres[f"test_{metric}"]))
            row[f"{metric}_std"] = float(np.std(cvres[f"test_{metric}"]))
        results.append(row)
        print(row)

results_df = pd.DataFrame(results)
print("\n=== CV comparison (train set, 5-fold) ===")
print(results_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Identify best strategy per model family (by macro-F1) and compare to baseline
# ---------------------------------------------------------------------------
summary_deltas = {}
for kind in model_kinds:
    sub = results_df[results_df.model == kind].set_index("imbalance_strategy")
    baseline_f1 = sub.loc["none", "f1_macro_mean"]
    baseline_auc = sub.loc["none", "roc_auc_mean"]
    best_strat = sub["f1_macro_mean"].idxmax()
    best_f1 = sub.loc[best_strat, "f1_macro_mean"]
    best_auc = sub.loc[best_strat, "roc_auc_mean"]
    summary_deltas[kind] = {
        "baseline_f1_macro": baseline_f1,
        "best_strategy": best_strat,
        "best_f1_macro": best_f1,
        "delta_f1_macro": best_f1 - baseline_f1,
        "baseline_roc_auc": baseline_auc,
        "best_roc_auc": best_auc,
        "delta_roc_auc": best_auc - baseline_auc,
    }
    print(f"\n[{kind}] baseline f1_macro={baseline_f1:.4f}, best={best_strat} "
          f"f1_macro={best_f1:.4f} (delta={best_f1-baseline_f1:+.4f}); "
          f"baseline AUC={baseline_auc:.4f}, best AUC={best_auc:.4f} "
          f"(delta={best_auc-baseline_auc:+.4f})")

# ---------------------------------------------------------------------------
# 6. Held-out test-set confirmation for Random Forest: baseline vs class_weight
#    (class_weight is the most standard / cheapest imbalance-handling approach;
#    we check whether it, and the CV-selected best strategy, replicate on test)
# ---------------------------------------------------------------------------
test_confirmation = {}
for kind in model_kinds:
    best_strat = summary_deltas[kind]["best_strategy"]
    for strat_label, strat in [("baseline_none", "none"), (f"best_{best_strat}", best_strat)]:
        pipe = make_model(kind, strat)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        test_confirmation[f"{kind}__{strat_label}"] = {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "f1_macro": float(f1_score(y_test, pred, average="macro")),
            "f1_minority": float(f1_score(y_test, pred)),
            "recall_minority": float(recall_score(y_test, pred)),
            "balanced_acc": float(balanced_accuracy_score(y_test, pred)),
        }

print("\n=== Held-out test set confirmation ===")
for k, v in test_confirmation.items():
    print(k, v)

# ---------------------------------------------------------------------------
# 7. Stability check: 5x repeated stratified 5-fold CV (different seeds),
#    Random Forest baseline vs class_weight='balanced', on the full training data.
# ---------------------------------------------------------------------------
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=123)

rf_none = make_model("rf", "none")
rf_cw = make_model("rf", "class_weight")

scores_none = cross_validate(rf_none, X_train, y_train, cv=rcv, scoring=scoring, n_jobs=-1)
scores_cw = cross_validate(rf_cw, X_train, y_train, cv=rcv, scoring=scoring, n_jobs=-1)

diff_f1 = scores_cw["test_f1_macro"] - scores_none["test_f1_macro"]
diff_auc = scores_cw["test_roc_auc"] - scores_none["test_roc_auc"]
diff_recall_min = scores_cw["test_recall_minority"] - scores_none["test_recall_minority"]

stability = {
    "n_folds_total": len(diff_f1),
    "f1_macro_diff_mean": float(np.mean(diff_f1)),
    "f1_macro_diff_std": float(np.std(diff_f1)),
    "f1_macro_diff_ci95": [
        float(np.mean(diff_f1) - 1.96 * np.std(diff_f1) / np.sqrt(len(diff_f1))),
        float(np.mean(diff_f1) + 1.96 * np.std(diff_f1) / np.sqrt(len(diff_f1))),
    ],
    "roc_auc_diff_mean": float(np.mean(diff_auc)),
    "roc_auc_diff_std": float(np.std(diff_auc)),
    "recall_minority_diff_mean": float(np.mean(diff_recall_min)),
    "recall_minority_diff_std": float(np.std(diff_recall_min)),
    "fraction_folds_f1_improved": float(np.mean(diff_f1 > 0)),
}

print("\n=== Stability check: RF baseline vs class_weight='balanced', "
      "5x repeated 5-fold CV (25 folds total) ===")
print(json.dumps(stability, indent=2))

# ---------------------------------------------------------------------------
# 8. Save everything needed for the report
# ---------------------------------------------------------------------------
output = {
    "cv_results_table": results_df.to_dict(orient="records"),
    "summary_deltas": summary_deltas,
    "test_confirmation": test_confirmation,
    "stability_check": stability,
}
with open("full_results.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nDone. Full results saved to full_results.json")
