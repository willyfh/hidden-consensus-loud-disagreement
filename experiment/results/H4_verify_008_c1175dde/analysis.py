"""
H4: Does addressing class imbalance improve model quality on the Adult Income dataset?

Approach
--------
1. Load and clean the data (strip whitespace, treat '?' as missing, drop rows with
   missing values in categorical columns since they are a small fraction).
2. Encode target as binary (1 = '>50K', 0 = '<=50K'). Check imbalance ratio.
3. Build a preprocessing pipeline (one-hot encode categoricals, scale numerics).
4. Train two model families (Logistic Regression, Random Forest) under three
   imbalance-handling regimes each:
     - baseline (no handling)
     - class_weight='balanced'
     - SMOTE oversampling on the training folds only
5. Evaluate with 5-fold stratified CV on the training split, using metrics that
   are informative under imbalance: ROC-AUC (threshold-free, imbalance-insensitive),
   PR-AUC / average precision (sensitive to minority-class performance), F1 and
   balanced accuracy at default 0.5 threshold, and recall on the minority class
   ('>50K'), since imbalance handling mainly trades precision for recall.
6. Refit best baseline vs best imbalance-handled model on full train, evaluate once
   on a held-out test set.
7. Stability check: repeated stratified CV (5 folds x 5 seeds = 25 runs) comparing
   baseline vs balanced/SMOTE variants, reporting mean +/- std of the metric deltas.
"""

import json
import warnings
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RepeatedStratifiedKFold, cross_validate
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score,
    recall_score, precision_score
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")
df.columns = [c.strip() for c in df.columns]

# strip whitespace from string columns, treat '?' as missing
for c in df.select_dtypes(include="object").columns:
    df[c] = df[c].str.strip()
    df[c] = df[c].replace("?", np.nan)

n_before = len(df)
df = df.dropna().reset_index(drop=True)
n_after = len(df)
print(f"Dropped {n_before - n_after} rows with missing values ({(n_before-n_after)/n_before:.2%})")

df["target"] = (df["class"].str.strip() == ">50K").astype(int)
df = df.drop(columns=["class"])

imbalance_ratio = df["target"].value_counts(normalize=True)
print("Class balance:\n", imbalance_ratio)

# drop education-num duplicate of education (keep education-num, drop education str) -- avoid redundant encoding
# Actually keep both is fine for tree/linear models but redundant category blowup; drop 'education' (string) and keep education-num
df = df.drop(columns=["education"])

# fnlwgt is a census sampling weight, not a real demographic feature; keep it as numeric predictor anyway (common in ML benchmarks)

target = df["target"]
X = df.drop(columns=["target"])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# ---------------------------------------------------------------------------
# 2. Train/test split (held out, untouched until final comparison)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, target, test_size=0.2, stratify=target, random_state=RANDOM_STATE
)
print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
print("Train class balance:", y_train.value_counts(normalize=True).to_dict())

# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

def make_pipeline(model, strategy):
    """strategy in {'baseline','balanced','smote'}"""
    if strategy == "smote":
        return ImbPipeline([
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ])
    else:
        return Pipeline([
            ("prep", preprocessor),
            ("model", model),
        ])

def get_model(name, strategy):
    if name == "logreg":
        cw = "balanced" if strategy == "balanced" else None
        return LogisticRegression(max_iter=1000, class_weight=cw, random_state=RANDOM_STATE)
    elif name == "rf":
        cw = "balanced" if strategy == "balanced" else None
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, min_samples_leaf=2,
            class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1
        )

scoring = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "balanced_acc": "balanced_accuracy",
    "recall_pos": "recall",
    "precision_pos": "precision",
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

results = []
for model_name in ["logreg", "rf"]:
    for strategy in ["baseline", "balanced", "smote"]:
        model = get_model(model_name, strategy)
        pipe = make_pipeline(model, strategy)
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        row = {"model": model_name, "strategy": strategy}
        for k in scoring:
            row[k + "_mean"] = cvres["test_" + k].mean()
            row[k + "_std"] = cvres["test_" + k].std()
        results.append(row)
        print(f"{model_name:7s} {strategy:9s} ROC-AUC={row['roc_auc_mean']:.4f} "
              f"PR-AUC={row['pr_auc_mean']:.4f} F1={row['f1_mean']:.4f} "
              f"BalAcc={row['balanced_acc_mean']:.4f} Recall+={row['recall_pos_mean']:.4f} "
              f"Prec+={row['precision_pos_mean']:.4f}")

results_df = pd.DataFrame(results)

# ---------------------------------------------------------------------------
# 4. Primary comparison: for each model family, baseline vs best imbalance-handling
#    strategy, using PR-AUC (most informative under imbalance) and balanced accuracy/F1
#    as the decision metrics. ROC-AUC is included as a threshold-free sanity check
#    (known to be largely insensitive to class-weighting/resampling in expectation).
# ---------------------------------------------------------------------------
print("\n=== CV Summary Table ===")
print(results_df[["model", "strategy", "roc_auc_mean", "pr_auc_mean", "f1_mean",
                   "balanced_acc_mean", "recall_pos_mean", "precision_pos_mean"]]
      .to_string(index=False))

# Compute deltas vs baseline per model family for F1 and balanced accuracy (primary
# "model quality" metrics for an imbalanced binary classification task) and PR-AUC.
deltas = []
for model_name in ["logreg", "rf"]:
    base = results_df[(results_df.model == model_name) & (results_df.strategy == "baseline")].iloc[0]
    for strategy in ["balanced", "smote"]:
        row = results_df[(results_df.model == model_name) & (results_df.strategy == strategy)].iloc[0]
        deltas.append({
            "model": model_name,
            "strategy": strategy,
            "d_f1": row["f1_mean"] - base["f1_mean"],
            "d_balanced_acc": row["balanced_acc_mean"] - base["balanced_acc_mean"],
            "d_pr_auc": row["pr_auc_mean"] - base["pr_auc_mean"],
            "d_roc_auc": row["roc_auc_mean"] - base["roc_auc_mean"],
            "d_recall_pos": row["recall_pos_mean"] - base["recall_pos_mean"],
            "d_precision_pos": row["precision_pos_mean"] - base["precision_pos_mean"],
        })
deltas_df = pd.DataFrame(deltas)
print("\n=== Deltas vs baseline (positive = imbalance-handling improved metric) ===")
print(deltas_df.to_string(index=False))

# Pick the primary finding: RandomForest, class_weight='balanced' vs baseline,
# since RF is typically the stronger base model on this dataset and 'balanced'
# class_weight is the most common practical imbalance fix.
# We'll determine best strategy per model by F1 (a standard single-number quality metric).
best_rows = results_df.loc[results_df.groupby("model")["f1_mean"].idxmax()]
print("\n=== Best strategy per model by F1 ===")
print(best_rows[["model", "strategy", "f1_mean", "balanced_acc_mean", "pr_auc_mean"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Held-out test evaluation: baseline vs balanced (class_weight) for both models
# ---------------------------------------------------------------------------
print("\n=== Held-out test set evaluation ===")
test_results = {}
for model_name in ["logreg", "rf"]:
    for strategy in ["baseline", "balanced", "smote"]:
        model = get_model(model_name, strategy)
        pipe = make_pipeline(model, strategy)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = pipe.predict(X_test)
        m = {
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "f1": f1_score(y_test, pred),
            "balanced_acc": balanced_accuracy_score(y_test, pred),
            "recall_pos": recall_score(y_test, pred),
            "precision_pos": precision_score(y_test, pred),
        }
        test_results[(model_name, strategy)] = m
        print(f"{model_name:7s} {strategy:9s} " + " ".join(f"{k}={v:.4f}" for k, v in m.items()))

# ---------------------------------------------------------------------------
# 6. Stability check: repeated stratified CV, 5 folds x 5 seeds, on TRAIN data,
#    comparing RF baseline vs RF balanced (the primary finding) via F1 and PR-AUC.
# ---------------------------------------------------------------------------
print("\n=== Stability check: Repeated 5-fold CV x 5 seeds (RF baseline vs RF balanced) ===")
rkf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=123)

rf_base_pipe = make_pipeline(get_model("rf", "baseline"), "baseline")
rf_bal_pipe = make_pipeline(get_model("rf", "balanced"), "balanced")

base_scores = cross_validate(rf_base_pipe, X_train, y_train, cv=rkf,
                              scoring=scoring, n_jobs=-1)
bal_scores = cross_validate(rf_bal_pipe, X_train, y_train, cv=rkf,
                             scoring=scoring, n_jobs=-1)

f1_deltas_rep = bal_scores["test_f1"] - base_scores["test_f1"]
pr_auc_deltas_rep = bal_scores["test_pr_auc"] - base_scores["test_pr_auc"]
bal_acc_deltas_rep = bal_scores["test_balanced_acc"] - base_scores["test_balanced_acc"]
roc_auc_deltas_rep = bal_scores["test_roc_auc"] - base_scores["test_roc_auc"]

print(f"RF baseline F1: {base_scores['test_f1'].mean():.4f} +/- {base_scores['test_f1'].std():.4f}")
print(f"RF balanced F1: {bal_scores['test_f1'].mean():.4f} +/- {bal_scores['test_f1'].std():.4f}")
print(f"F1 delta (balanced - baseline): mean={f1_deltas_rep.mean():.4f}, std={f1_deltas_rep.std():.4f}, "
      f"min={f1_deltas_rep.min():.4f}, max={f1_deltas_rep.max():.4f}")
print(f"PR-AUC delta: mean={pr_auc_deltas_rep.mean():.4f}, std={pr_auc_deltas_rep.std():.4f}")
print(f"Balanced-Accuracy delta: mean={bal_acc_deltas_rep.mean():.4f}, std={bal_acc_deltas_rep.std():.4f}")
print(f"ROC-AUC delta: mean={roc_auc_deltas_rep.mean():.4f}, std={roc_auc_deltas_rep.std():.4f}")

n_positive_f1 = (f1_deltas_rep > 0).sum()
n_total = len(f1_deltas_rep)
print(f"F1 improved in {n_positive_f1}/{n_total} of the 25 repeated CV runs")

# 95% CI via normal approx on the 25 repeated-CV delta estimates (paired, same folds)
from scipy import stats
ci_low, ci_high = stats.t.interval(
    0.95, df=len(f1_deltas_rep) - 1,
    loc=f1_deltas_rep.mean(), scale=stats.sem(f1_deltas_rep)
)
print(f"95% CI for F1 delta (balanced - baseline): [{ci_low:.4f}, {ci_high:.4f}]")

bal_acc_ci_low, bal_acc_ci_high = stats.t.interval(
    0.95, df=len(bal_acc_deltas_rep) - 1,
    loc=bal_acc_deltas_rep.mean(), scale=stats.sem(bal_acc_deltas_rep)
)
print(f"95% CI for Balanced-Accuracy delta: [{bal_acc_ci_low:.4f}, {bal_acc_ci_high:.4f}]")

# ---------------------------------------------------------------------------
# Save summary artifacts
# ---------------------------------------------------------------------------
summary = {
    "n_rows_used": int(n_after),
    "n_dropped_missing": int(n_before - n_after),
    "class_balance_full": imbalance_ratio.to_dict(),
    "cv_results": results_df.to_dict(orient="records"),
    "deltas_vs_baseline": deltas_df.to_dict(orient="records"),
    "test_results": {f"{k[0]}_{k[1]}": v for k, v in test_results.items()},
    "repeated_cv_f1_delta_mean": float(f1_deltas_rep.mean()),
    "repeated_cv_f1_delta_std": float(f1_deltas_rep.std()),
    "repeated_cv_f1_delta_ci95": [float(ci_low), float(ci_high)],
    "repeated_cv_balanced_acc_delta_mean": float(bal_acc_deltas_rep.mean()),
    "repeated_cv_balanced_acc_delta_ci95": [float(bal_acc_ci_low), float(bal_acc_ci_high)],
    "repeated_cv_pr_auc_delta_mean": float(pr_auc_deltas_rep.mean()),
    "repeated_cv_roc_auc_delta_mean": float(roc_auc_deltas_rep.mean()),
    "n_positive_f1_out_of_25": int(n_positive_f1),
}
with open("cv_summary.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)

print("\nDone. Summary written to cv_summary.json")
