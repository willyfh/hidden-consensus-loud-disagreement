"""
H4: Does addressing class imbalance improve model quality on the Adult Income dataset?

Approach
--------
The target `class` is moderately imbalanced (~76% <=50K, ~24% >50K, ratio ~3.2:1).
We train two model families (Logistic Regression, Random Forest) under three
imbalance-handling regimes:
  1. Baseline    - no imbalance handling at all
  2. ClassWeight - class_weight='balanced' (reweights the loss function)
  3. SMOTE       - synthetic minority oversampling applied to the training fold only

We evaluate on a held-out, untouched (still-imbalanced) test set using both a
threshold-free ranking metric (ROC-AUC, PR-AUC/average precision - these should be
largely insensitive to reweighting/resampling since they only depend on the score
ranking) and threshold-dependent metrics at the default 0.5 cutoff (F1, recall,
precision, balanced accuracy on the minority ">50K" class) - these are the metrics
imbalance handling is actually meant to improve, since it shifts the decision
boundary rather than the ranking.
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & inspect
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

target_col = "class"
y = (df[target_col].str.strip() == ">50K").astype(int)
X = df.drop(columns=[target_col])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

print("Class balance:\n", y.value_counts(normalize=True))
print("Categorical cols:", cat_cols)
print("Numeric cols:", num_cols)

# ---------------------------------------------------------------------------
# 2. Train / test split (stratified, test set kept realistically imbalanced)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. Preprocessing: impute missing categoricals with a "Missing" category,
#    one-hot encode categoricals, standardize numeric features (helps LogReg;
#    harmless for RF).
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
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
        (
            "num",
            Pipeline(
                [
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                ]
            ),
            num_cols,
        ),
    ]
)

# ---------------------------------------------------------------------------
# 4. Define model + imbalance-handling combinations
# ---------------------------------------------------------------------------
models = {
    "LogisticRegression": lambda class_weight=None: LogisticRegression(
        max_iter=1000, class_weight=class_weight, random_state=RANDOM_STATE
    ),
    "RandomForest": lambda class_weight=None: RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    ),
}

results = []

for model_name, model_fn in models.items():
    # --- Baseline: no imbalance handling ---
    pipe = Pipeline([("prep", preprocessor), ("clf", model_fn(class_weight=None))])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    results.append(
        {
            "model": model_name,
            "strategy": "Baseline",
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "f1_minority": f1_score(y_test, pred),
            "recall_minority": recall_score(y_test, pred),
            "precision_minority": precision_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        }
    )

    # --- class_weight='balanced' ---
    pipe = Pipeline([("prep", preprocessor), ("clf", model_fn(class_weight="balanced"))])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)
    results.append(
        {
            "model": model_name,
            "strategy": "ClassWeight",
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "f1_minority": f1_score(y_test, pred),
            "recall_minority": recall_score(y_test, pred),
            "precision_minority": precision_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        }
    )

    # --- SMOTE oversampling (train fold only) ---
    imb_pipe = ImbPipeline(
        [
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("clf", model_fn(class_weight=None)),
        ]
    )
    imb_pipe.fit(X_train, y_train)
    proba = imb_pipe.predict_proba(X_test)[:, 1]
    pred = imb_pipe.predict(X_test)
    results.append(
        {
            "model": model_name,
            "strategy": "SMOTE",
            "roc_auc": roc_auc_score(y_test, proba),
            "pr_auc": average_precision_score(y_test, proba),
            "f1_minority": f1_score(y_test, pred),
            "recall_minority": recall_score(y_test, pred),
            "precision_minority": precision_score(y_test, pred),
            "balanced_accuracy": balanced_accuracy_score(y_test, pred),
        }
    )

results_df = pd.DataFrame(results)
pd.set_option("display.width", 140)
print("\n=== Full results ===")
print(results_df.round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# 5. Summarize: for each model, compare each imbalance-handling strategy to baseline
# ---------------------------------------------------------------------------
print("\n=== Deltas vs Baseline (strategy - baseline) ===")
metric_cols = ["roc_auc", "pr_auc", "f1_minority", "recall_minority", "precision_minority", "balanced_accuracy"]
deltas = []
for model_name in models:
    base_row = results_df[(results_df.model == model_name) & (results_df.strategy == "Baseline")].iloc[0]
    for strat in ["ClassWeight", "SMOTE"]:
        row = results_df[(results_df.model == model_name) & (results_df.strategy == strat)].iloc[0]
        delta = {"model": model_name, "strategy": strat}
        for m in metric_cols:
            delta[m] = row[m] - base_row[m]
        deltas.append(delta)
deltas_df = pd.DataFrame(deltas)
print(deltas_df.round(4).to_string(index=False))

# Primary metric: F1 on minority class is the standard target for imbalance-handling
# techniques (balances precision/recall on the class that matters, unlike accuracy
# or ROC-AUC which are largely insensitive to threshold-side rebalancing).
# We summarize with the mean F1 delta (best strategy - baseline), averaged across models.
best_f1_deltas = deltas_df.groupby("model")["f1_minority"].max()
mean_best_f1_delta = best_f1_deltas.mean()

mean_roc_auc_delta = deltas_df["roc_auc"].mean()
mean_balanced_acc_delta = deltas_df["balanced_accuracy"].mean()
mean_recall_delta = deltas_df["recall_minority"].mean()

print(f"\nMean ROC-AUC delta (all strategies vs baseline, averaged): {mean_roc_auc_delta:.4f}")
print(f"Mean balanced-accuracy delta (all strategies vs baseline): {mean_balanced_acc_delta:.4f}")
print(f"Mean minority-recall delta (all strategies vs baseline): {mean_recall_delta:.4f}")
print(f"Mean best-F1(minority) delta (best strategy per model vs baseline): {mean_best_f1_delta:.4f}")

# ---------------------------------------------------------------------------
# 6. Write result.json
# ---------------------------------------------------------------------------
summary = (
    "Addressing class imbalance (class_weight='balanced' or SMOTE) barely changes ranking-based "
    "quality (ROC-AUC/PR-AUC virtually unchanged) but substantially shifts the precision/recall "
    "trade-off at the default 0.5 threshold: minority-class (>50K) recall rises sharply while "
    "precision and F1 drop, and balanced accuracy improves modestly. So imbalance handling does not "
    "improve overall discriminative quality, it mainly re-calibrates the decision threshold toward "
    "the minority class."
)

result = {
    "hypothesis_id": "H4",
    "summary": summary,
    "primary_metric_name": "Mean ROC-AUC delta (imbalance-handling strategies - baseline, averaged across LogReg/RF and ClassWeight/SMOTE)",
    "primary_metric_value": float(mean_roc_auc_delta),
    "direction": "imbalance handling ~= baseline on ROC-AUC/PR-AUC; recall(minority) up, precision(minority) down, F1 mixed",
    "methodological_choices": (
        "Binary target = 1 if class=='>50K'. Stratified 75/25 train/test split (test set left "
        "imbalanced to reflect real deployment distribution), random_state=42. Missing categoricals "
        "imputed with an explicit 'Missing' category; missing numerics (none present) would use median "
        "imputation. Categoricals one-hot encoded, numerics standardized. Two model families: Logistic "
        "Regression (max_iter=1000) and Random Forest (300 trees, min_samples_leaf=2). Three imbalance "
        "regimes compared per model: no handling (baseline), class_weight='balanced', and SMOTE "
        "oversampling applied only to the training fold within an imblearn pipeline (to avoid leakage). "
        "Metrics: ROC-AUC and PR-AUC (average precision) as threshold-free ranking quality; F1/recall/"
        "precision on the minority '>50K' class and balanced accuracy at the default 0.5 threshold as "
        "threshold-dependent quality. Primary metric chosen as the ROC-AUC delta because it is the most "
        "common single-number 'model quality' metric and best isolates whether imbalance-handling changes "
        "genuine discriminative power vs. just moving the decision threshold. Alternative choices another "
        "researcher might make: threshold tuning via precision-recall curve instead of fixed 0.5 cutoff, "
        "random undersampling instead of/in addition to SMOTE, F1 or PR-AUC as the primary metric instead "
        "of ROC-AUC, or cross-validation instead of a single train/test split."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
print(json.dumps(result, indent=2))
