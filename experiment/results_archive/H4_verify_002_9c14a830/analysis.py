"""
H4: Does addressing class imbalance improve model quality on the Adult Income dataset?

Approach:
- Load and clean adult_income.csv (target `class`: <=50K / >50K, ~76%/24% split -> moderate imbalance).
- Preprocess: median/mode imputation for missing categoricals (kept as explicit 'Missing' category),
  one-hot encoding for categorical features, standard scaling for numeric features (for LogReg).
- Models: Logistic Regression and Random Forest, each evaluated in three imbalance-handling variants:
    (a) baseline - no imbalance handling
    (b) class_weight='balanced'
    (c) SMOTE oversampling of the training fold only
- Single 70/30 stratified train/test split for the primary comparison.
- Metrics: ROC-AUC, PR-AUC (average precision), balanced accuracy, F1 (minority/">50K" class), MCC.
  ROC-AUC/PR-AUC are threshold-free ranking metrics (expected to be roughly stable under reweighting);
  balanced accuracy, minority F1 and MCC are threshold-dependent (0.5 cutoff) and are where imbalance
  handling is expected to matter most.
- Stability check: 5x repeated stratified 5-fold CV (5 different seeds) on the full dataset, comparing
  baseline vs class-weighted Random Forest on balanced accuracy and minority-class F1.
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
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
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

target_col = "class"
y = (df[target_col].astype(str).str.strip() == ">50K").astype(int)  # 1 = minority (>50K)

X = df.drop(columns=[target_col])

cat_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Fill missing categoricals with explicit "Missing" category
for c in cat_cols:
    X[c] = X[c].fillna("Missing")

print("Rows:", len(X), "| Positive rate (>50K):", y.mean().round(4))
print("Categorical cols:", cat_cols)
print("Numeric cols:", num_cols)

# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# ---------------------------------------------------------------------------
# 3. Primary comparison: single 70/30 stratified split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

def make_pipeline(estimator, imbalance_method):
    if imbalance_method == "smote":
        return ImbPipeline(
            steps=[
                ("prep", preprocessor),
                ("smote", SMOTE(random_state=RANDOM_STATE)),
                ("clf", estimator),
            ]
        )
    return Pipeline(steps=[("prep", preprocessor), ("clf", estimator)])


def get_estimator(model_name, imbalance_method):
    if model_name == "logreg":
        cw = "balanced" if imbalance_method == "class_weight" else None
        return LogisticRegression(max_iter=2000, class_weight=cw, random_state=RANDOM_STATE)
    elif model_name == "rf":
        cw = "balanced" if imbalance_method == "class_weight" else None
        return RandomForestClassifier(
            n_estimators=300, max_depth=None, class_weight=cw,
            random_state=RANDOM_STATE, n_jobs=-1
        )
    raise ValueError(model_name)


def evaluate(pipeline, X_tr, y_tr, X_te, y_te):
    pipeline.fit(X_tr, y_tr)
    proba = pipeline.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "f1_minority": f1_score(y_te, pred, pos_label=1),
        "mcc": matthews_corrcoef(y_te, pred),
    }


results = {}
for model_name in ["logreg", "rf"]:
    for imbalance_method in ["none", "class_weight", "smote"]:
        est = get_estimator(model_name, imbalance_method)
        pipe = make_pipeline(est, imbalance_method)
        metrics = evaluate(pipe, X_train, y_train, X_test, y_test)
        key = f"{model_name}__{imbalance_method}"
        results[key] = metrics
        print(key, metrics)

print("\n=== Primary comparison (single 70/30 split) ===")
for k, v in results.items():
    print(k, {m: round(val, 4) for m, val in v.items()})

# ---------------------------------------------------------------------------
# 4. Compute deltas (imbalance-handled minus baseline) for RF (primary model)
# ---------------------------------------------------------------------------
base_rf = results["rf__none"]
cw_rf = results["rf__class_weight"]
smote_rf = results["rf__smote"]

delta_cw_rf = {m: cw_rf[m] - base_rf[m] for m in base_rf}
delta_smote_rf = {m: smote_rf[m] - base_rf[m] for m in base_rf}

base_lr = results["logreg__none"]
cw_lr = results["logreg__class_weight"]
smote_lr = results["logreg__smote"]
delta_cw_lr = {m: cw_lr[m] - base_lr[m] for m in base_lr}
delta_smote_lr = {m: smote_lr[m] - base_lr[m] for m in base_lr}

print("\nRF delta (class_weight - none):", {k: round(v, 4) for k, v in delta_cw_rf.items()})
print("RF delta (smote - none):", {k: round(v, 4) for k, v in delta_smote_rf.items()})
print("LogReg delta (class_weight - none):", {k: round(v, 4) for k, v in delta_cw_lr.items()})
print("LogReg delta (smote - none):", {k: round(v, 4) for k, v in delta_smote_lr.items()})

# ---------------------------------------------------------------------------
# 5. Stability check: 5x repeated stratified 5-fold CV, RF baseline vs class_weight
# ---------------------------------------------------------------------------
print("\n=== Stability check: 5x5 repeated stratified CV (RF: none vs class_weight) ===")

rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=RANDOM_STATE)

cv_records = {"none": {"balanced_accuracy": [], "f1_minority": [], "roc_auc": []},
              "class_weight": {"balanced_accuracy": [], "f1_minority": [], "roc_auc": []}}

for fold_i, (tr_idx, te_idx) in enumerate(rskf.split(X, y)):
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    for imbalance_method in ["none", "class_weight"]:
        est = get_estimator("rf", imbalance_method)
        pipe = make_pipeline(est, imbalance_method)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        pred = (proba >= 0.5).astype(int)
        cv_records[imbalance_method]["balanced_accuracy"].append(balanced_accuracy_score(y_te, pred))
        cv_records[imbalance_method]["f1_minority"].append(f1_score(y_te, pred, pos_label=1))
        cv_records[imbalance_method]["roc_auc"].append(roc_auc_score(y_te, proba))
    if (fold_i + 1) % 5 == 0:
        print(f"  completed {fold_i + 1}/25 folds")

cv_summary = {}
for method in ["none", "class_weight"]:
    cv_summary[method] = {
        m: {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        for m, vals in cv_records[method].items()
    }

print("\nCV summary:")
for method, metrics in cv_summary.items():
    print(method, {m: {k: round(v, 4) for k, v in d.items()} for m, d in metrics.items()})

ba_diffs = np.array(cv_records["class_weight"]["balanced_accuracy"]) - np.array(cv_records["none"]["balanced_accuracy"])
f1_diffs = np.array(cv_records["class_weight"]["f1_minority"]) - np.array(cv_records["none"]["f1_minority"])
auc_diffs = np.array(cv_records["class_weight"]["roc_auc"]) - np.array(cv_records["none"]["roc_auc"])

print("\nCV delta (class_weight - none) balanced_accuracy: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
    ba_diffs.mean(), ba_diffs.std(), ba_diffs.min(), ba_diffs.max()))
print("CV delta (class_weight - none) f1_minority: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
    f1_diffs.mean(), f1_diffs.std(), f1_diffs.min(), f1_diffs.max()))
print("CV delta (class_weight - none) roc_auc: mean={:.4f} std={:.4f} min={:.4f} max={:.4f}".format(
    auc_diffs.mean(), auc_diffs.std(), auc_diffs.min(), auc_diffs.max()))

# ---------------------------------------------------------------------------
# 6. Save results
# ---------------------------------------------------------------------------
output = {
    "single_split_results": results,
    "rf_delta_class_weight_minus_none": delta_cw_rf,
    "rf_delta_smote_minus_none": delta_smote_rf,
    "logreg_delta_class_weight_minus_none": delta_cw_lr,
    "logreg_delta_smote_minus_none": delta_smote_lr,
    "cv_summary": cv_summary,
    "cv_balanced_accuracy_diff": {"mean": float(ba_diffs.mean()), "std": float(ba_diffs.std())},
    "cv_f1_minority_diff": {"mean": float(f1_diffs.mean()), "std": float(f1_diffs.std())},
    "cv_roc_auc_diff": {"mean": float(auc_diffs.mean()), "std": float(auc_diffs.std())},
}

with open("analysis_output.json", "w") as f:
    json.dump(output, f, indent=2)

print("\nSaved analysis_output.json")
