"""
H4: Does addressing class imbalance improve model quality on the Adult Income dataset?

Approach
--------
1. Load & clean data (missing values coded as '?').
2. Encode categoricals (OneHot for linear model, ordinal-safe OneHot for tree model too --
   we use OneHotEncoding for both for consistency; RF handles it fine).
3. Split into train/test (stratified, 70/30), fixed seed for the primary run.
4. Establish class imbalance ratio.
5. Train two model families (Logistic Regression, Random Forest) under three imbalance-handling
   strategies each:
     (a) baseline (no imbalance handling)
     (b) class_weight='balanced'
     (c) SMOTE oversampling on the training fold only
6. Evaluate on the untouched test set using metrics that matter under imbalance:
   ROC-AUC (threshold-free, primary metric), PR-AUC (average precision), balanced accuracy,
   F1 (minority class), recall (minority class), precision (minority class).
7. Compare baseline vs. imbalance-handled variants. Since ROC-AUC is threshold independent and
   fairly insensitive to imbalance handling for well-calibrated probabilistic models, we treat
   it as a sanity check, but the primary metric for "model quality under imbalance" is
   macro-F1 / minority-class F1 and balanced accuracy, which directly capture whether the
   minority class (>50K, ~24% of data) is being served well.
8. Stability check: 5x repeated stratified 5-fold CV (5 different seeds) comparing baseline vs
   class-weight-balanced Logistic Regression and Random Forest on macro-F1 and ROC-AUC,
   plus a bootstrap CI on the test-set difference.
"""

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
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

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")
df = df.replace("?", np.nan)
df = df.dropna().reset_index(drop=True)

target_col = "class"
y = (df[target_col].str.strip() == ">50K").astype(int)
X = df.drop(columns=[target_col])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

imbalance_ratio = y.value_counts(normalize=True).to_dict()
print("Rows after dropping missing:", len(df))
print("Class distribution (1 = >50K):", imbalance_ratio)

# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# ---------------------------------------------------------------------------
# 3. Train/test split (primary run)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

print("\nTrain class balance:", y_train.value_counts(normalize=True).to_dict())
print("Test class balance:", y_test.value_counts(normalize=True).to_dict())


def make_pipeline(model_name, strategy):
    """Build a pipeline for a given model_name in {'logreg','rf'} and
    strategy in {'baseline','class_weight','smote'}."""
    if model_name == "logreg":
        if strategy == "class_weight":
            clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE)
        else:
            clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    elif model_name == "rf":
        if strategy == "class_weight":
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=None, class_weight="balanced",
                random_state=RANDOM_STATE, n_jobs=-1
            )
        else:
            clf = RandomForestClassifier(
                n_estimators=300, max_depth=None, random_state=RANDOM_STATE, n_jobs=-1
            )
    else:
        raise ValueError(model_name)

    if strategy == "smote":
        pipe = ImbPipeline(steps=[
            ("prep", preprocess),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", clf),
        ])
    else:
        pipe = Pipeline(steps=[
            ("prep", preprocess),
            ("clf", clf),
        ])
    return pipe


def evaluate(pipe, X_tr, y_tr, X_te, y_te):
    pipe.fit(X_tr, y_tr)
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = pipe.predict(X_te)
    return {
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "f1_minority": f1_score(y_te, pred, pos_label=1),
        "precision_minority": precision_score(y_te, pred, pos_label=1),
        "recall_minority": recall_score(y_te, pred, pos_label=1),
        "macro_f1": f1_score(y_te, pred, average="macro"),
    }


results = {}
for model_name in ["logreg", "rf"]:
    for strategy in ["baseline", "class_weight", "smote"]:
        pipe = make_pipeline(model_name, strategy)
        metrics = evaluate(pipe, X_train, y_train, X_test, y_test)
        results[f"{model_name}__{strategy}"] = metrics
        print(f"\n{model_name} / {strategy}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

# ---------------------------------------------------------------------------
# Summarize deltas (imbalance-handled minus baseline) per model
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("DELTAS vs baseline (positive = improvement)")
print("=" * 70)
deltas = {}
for model_name in ["logreg", "rf"]:
    base = results[f"{model_name}__baseline"]
    for strategy in ["class_weight", "smote"]:
        strat_metrics = results[f"{model_name}__{strategy}"]
        d = {k: strat_metrics[k] - base[k] for k in base}
        deltas[f"{model_name}__{strategy}"] = d
        print(f"\n{model_name} / {strategy} - baseline:")
        for k, v in d.items():
            print(f"  {k}: {v:+.4f}")

# ---------------------------------------------------------------------------
# 6. Stability check: repeated stratified CV, 5 seeds x 5 folds
#    Compare baseline vs class_weight (fast, deterministic) for both models
#    on the FULL X (not just train) using macro_f1 and roc_auc.
#    Also run SMOTE variant via CV (slower, so use 5 folds x 3 seeds for SMOTE to keep runtime reasonable).
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STABILITY CHECK: Repeated Stratified CV")
print("=" * 70)

cv_results = {}
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

scoring = {
    "roc_auc": "roc_auc",
    "macro_f1": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
}

for model_name in ["logreg", "rf"]:
    for strategy in ["baseline", "class_weight"]:
        pipe = make_pipeline(model_name, strategy)
        cvres = cross_validate(pipe, X, y, cv=rskf, scoring=scoring, n_jobs=-1)
        cv_results[f"{model_name}__{strategy}"] = {
            k: (float(np.mean(v)), float(np.std(v)))
            for k, v in cvres.items() if k.startswith("test_")
        }
        print(f"\n{model_name} / {strategy} (5x5 repeated CV):")
        for k, (m, s) in cv_results[f"{model_name}__{strategy}"].items():
            print(f"  {k}: {m:.4f} +/- {s:.4f}")

# SMOTE via CV -- 5 folds x 3 repeats (SMOTE is heavier)
rskf_smote = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)
for model_name in ["logreg", "rf"]:
    pipe = make_pipeline(model_name, "smote")
    cvres = cross_validate(pipe, X, y, cv=rskf_smote, scoring=scoring, n_jobs=-1)
    cv_results[f"{model_name}__smote"] = {
        k: (float(np.mean(v)), float(np.std(v)))
        for k, v in cvres.items() if k.startswith("test_")
    }
    print(f"\n{model_name} / smote (5x3 repeated CV):")
    for k, (m, s) in cv_results[f"{model_name}__smote"].items():
        print(f"  {k}: {m:.4f} +/- {s:.4f}")

# ---------------------------------------------------------------------------
# Compute CV deltas
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("CV DELTAS vs baseline (positive = improvement)")
print("=" * 70)
cv_deltas = {}
for model_name in ["logreg", "rf"]:
    base = cv_results[f"{model_name}__baseline"]
    for strategy in ["class_weight", "smote"]:
        strat = cv_results[f"{model_name}__{strategy}"]
        d = {k: strat[k][0] - base[k][0] for k in base}
        cv_deltas[f"{model_name}__{strategy}"] = d
        print(f"\n{model_name} / {strategy} - baseline (CV mean delta):")
        for k, v in d.items():
            print(f"  {k}: {v:+.4f}")

# ---------------------------------------------------------------------------
# Bootstrap CI on test-set macro_f1 delta for the most informative comparison:
# RF baseline vs RF class_weight (tree model, class_weight strategy, on held-out test set)
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("BOOTSTRAP CI: RF baseline vs RF class_weight, test-set macro F1 delta")
print("=" * 70)

pipe_base = make_pipeline("rf", "baseline")
pipe_base.fit(X_train, y_train)
pred_base = pipe_base.predict(X_test)

pipe_cw = make_pipeline("rf", "class_weight")
pipe_cw.fit(X_train, y_train)
pred_cw = pipe_cw.predict(X_test)

y_test_arr = y_test.reset_index(drop=True).values
pred_base_arr = np.array(pred_base)
pred_cw_arr = np.array(pred_cw)

rng = np.random.default_rng(RANDOM_STATE)
n = len(y_test_arr)
n_boot = 2000
boot_deltas = np.empty(n_boot)
for i in range(n_boot):
    idx = rng.integers(0, n, n)
    f1_base = f1_score(y_test_arr[idx], pred_base_arr[idx], average="macro")
    f1_cw = f1_score(y_test_arr[idx], pred_cw_arr[idx], average="macro")
    boot_deltas[i] = f1_cw - f1_base

ci_low, ci_high = np.percentile(boot_deltas, [2.5, 97.5])
point_delta = f1_score(y_test_arr, pred_cw_arr, average="macro") - f1_score(y_test_arr, pred_base_arr, average="macro")
print(f"Point estimate (macro F1 delta, class_weight - baseline): {point_delta:+.4f}")
print(f"95% Bootstrap CI: [{ci_low:+.4f}, {ci_high:+.4f}]")
print(f"CI excludes zero: {ci_low > 0 or ci_high < 0}")

# ---------------------------------------------------------------------------
# Save everything needed for result.json
# ---------------------------------------------------------------------------
output = {
    "imbalance_ratio": imbalance_ratio,
    "test_results": results,
    "test_deltas": deltas,
    "cv_results": cv_results,
    "cv_deltas": cv_deltas,
    "bootstrap": {
        "point_delta_macro_f1_rf_classweight_minus_baseline": float(point_delta),
        "ci_95_low": float(ci_low),
        "ci_95_high": float(ci_high),
    },
}

with open("analysis_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved detailed output to analysis_output.json")
