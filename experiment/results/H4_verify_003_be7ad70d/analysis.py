"""
H4: Does addressing class imbalance improve model quality on the Adult Income dataset?

Approach
--------
1. Load and clean the data (missing values coded as "?").
2. One-hot encode categoricals, scale numerics.
3. Split into train/test (stratified, 75/25).
4. Fit two models per algorithm (Logistic Regression, Random Forest):
   - "baseline": no imbalance handling.
   - "balanced": class_weight="balanced" (reweights the loss inversely
     proportional to class frequency -- a standard, cheap way of "addressing
     imbalance" that doesn't require synthetic sampling libraries).
5. Compare using metrics that are informative under imbalance: ROC-AUC,
   PR-AUC (average precision), balanced accuracy, macro-F1, and per-class
   recall. Plain accuracy is reported too but is not the deciding metric
   since the majority class dominates it.
6. Validate the primary finding (does class_weight="balanced" change
   quality, and in which direction) with 5x repeated stratified 5-fold CV
   using different seeds on the training data, plus a re-check on a fresh
   held-out split.
"""

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    f1_score, recall_score, accuracy_score
)

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")
df = df.replace("?", np.nan)

target_col = "class"
y = (df[target_col].str.strip() == ">50K").astype(int)
X = df.drop(columns=[target_col])

print("Rows:", len(df))
print("Class distribution:\n", df[target_col].value_counts(normalize=True))

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()
print("Categorical cols:", cat_cols)
print("Numeric cols:", num_cols)
print("Missing values per column:\n", X.isna().sum()[X.isna().sum() > 0])

# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------
preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
])

# ---------------------------------------------------------------------------
# 3. Train/test split (held out for final evaluation)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

models = {
    "LogReg_baseline": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "LogReg_balanced": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"),
    "RF_baseline": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
    "RF_balanced": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"),
}

def evaluate(pipe, X_te, y_te):
    proba = pipe.predict_proba(X_te)[:, 1]
    pred = pipe.predict(X_te)
    return {
        "roc_auc": roc_auc_score(y_te, proba),
        "pr_auc": average_precision_score(y_te, proba),
        "balanced_accuracy": balanced_accuracy_score(y_te, pred),
        "macro_f1": f1_score(y_te, pred, average="macro"),
        "recall_minority(>50K)": recall_score(y_te, pred, pos_label=1),
        "recall_majority(<=50K)": recall_score(y_te, pred, pos_label=0),
        "accuracy": accuracy_score(y_te, pred),
    }

results = {}
fitted = {}
for name, clf in models.items():
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    pipe.fit(X_train, y_train)
    fitted[name] = pipe
    results[name] = evaluate(pipe, X_test, y_test)

print("\n=== Held-out test set results ===")
res_df = pd.DataFrame(results).T
print(res_df.round(4))

# Primary comparison: does class_weight="balanced" improve quality?
# Use ROC-AUC (threshold-independent, standard for imbalance) and macro-F1
# (threshold-dependent, sensitive to minority-class performance) as the
# headline pair; ROC-AUC is chosen as THE primary metric since it doesn't
# depend on the default 0.5 decision threshold, which is the main thing
# class_weight="balanced" shifts.
logreg_roc_delta = results["LogReg_balanced"]["roc_auc"] - results["LogReg_baseline"]["roc_auc"]
rf_roc_delta = results["RF_balanced"]["roc_auc"] - results["RF_baseline"]["roc_auc"]
logreg_f1_delta = results["LogReg_balanced"]["macro_f1"] - results["LogReg_baseline"]["macro_f1"]
rf_f1_delta = results["RF_balanced"]["macro_f1"] - results["RF_baseline"]["macro_f1"]

print(f"\nLogReg ROC-AUC delta (balanced - baseline): {logreg_roc_delta:.4f}")
print(f"RF ROC-AUC delta (balanced - baseline): {rf_roc_delta:.4f}")
print(f"LogReg macro-F1 delta (balanced - baseline): {logreg_f1_delta:.4f}")
print(f"RF macro-F1 delta (balanced - baseline): {rf_f1_delta:.4f}")

# ---------------------------------------------------------------------------
# 4. Stability check: 5x repeated 5-fold CV on the training data with
#    different seeds, comparing RF baseline vs RF balanced (RF is the
#    stronger model of the two, so it is the one we validate).
# ---------------------------------------------------------------------------
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=1)

scoring = {
    "roc_auc": "roc_auc",
    "macro_f1": "f1_macro",
    "balanced_accuracy": "balanced_accuracy",
}

cv_results = {}
for name in ["RF_baseline", "RF_balanced", "LogReg_baseline", "LogReg_balanced"]:
    clf = models[name].__class__(**models[name].get_params())
    pipe = Pipeline([("prep", preprocess), ("clf", clf)])
    cv = cross_validate(pipe, X_train, y_train, cv=rskf, scoring=scoring, n_jobs=-1)
    cv_results[name] = {
        "roc_auc_mean": cv["test_roc_auc"].mean(),
        "roc_auc_std": cv["test_roc_auc"].std(),
        "macro_f1_mean": cv["test_macro_f1"].mean(),
        "macro_f1_std": cv["test_macro_f1"].std(),
        "balanced_accuracy_mean": cv["test_balanced_accuracy"].mean(),
        "balanced_accuracy_std": cv["test_balanced_accuracy"].std(),
    }

cv_df = pd.DataFrame(cv_results).T
print("\n=== 5x5 repeated stratified CV (training data) ===")
print(cv_df.round(4))

rf_cv_roc_delta = cv_results["RF_balanced"]["roc_auc_mean"] - cv_results["RF_baseline"]["roc_auc_mean"]
rf_cv_f1_delta = cv_results["RF_balanced"]["macro_f1_mean"] - cv_results["RF_baseline"]["macro_f1_mean"]
logreg_cv_roc_delta = cv_results["LogReg_balanced"]["roc_auc_mean"] - cv_results["LogReg_baseline"]["roc_auc_mean"]
logreg_cv_f1_delta = cv_results["LogReg_balanced"]["macro_f1_mean"] - cv_results["LogReg_baseline"]["macro_f1_mean"]

print(f"\nCV RF ROC-AUC delta (balanced - baseline): {rf_cv_roc_delta:.4f}")
print(f"CV RF macro-F1 delta (balanced - baseline): {rf_cv_f1_delta:.4f}")
print(f"CV LogReg ROC-AUC delta (balanced - baseline): {logreg_cv_roc_delta:.4f}")
print(f"CV LogReg macro-F1 delta (balanced - baseline): {logreg_cv_f1_delta:.4f}")

# ---------------------------------------------------------------------------
# 5. Save all numbers for reporting
# ---------------------------------------------------------------------------
summary = {
    "test_set_results": results,
    "cv_results": cv_results,
    "test_set_deltas": {
        "logreg_roc_auc_delta": logreg_roc_delta,
        "rf_roc_auc_delta": rf_roc_delta,
        "logreg_macro_f1_delta": logreg_f1_delta,
        "rf_macro_f1_delta": rf_f1_delta,
    },
    "cv_deltas": {
        "rf_roc_auc_delta": rf_cv_roc_delta,
        "rf_macro_f1_delta": rf_cv_f1_delta,
        "logreg_roc_auc_delta": logreg_cv_roc_delta,
        "logreg_macro_f1_delta": logreg_cv_f1_delta,
    },
}

with open("analysis_details.json", "w") as f:
    json.dump(summary, f, indent=2, default=float)

print("\nDone. Details written to analysis_details.json")
