"""
H1: Does the choice of model family meaningfully affect predictive performance
on the Adult (Census Income) dataset?

Independent analysis. Trains several distinct model families under an
identical preprocessing pipeline and identical train/test split, then
compares held-out performance (ROC-AUC, plus accuracy/F1/PR-AUC) and
5-fold cross-validated ROC-AUC for stability.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, average_precision_score

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

# Target: binarize class label
y = (df["class"].str.strip() == ">50K").astype(int)

# Drop redundant / non-predictive columns:
#   - 'class' is the target itself
#   - 'education' is a string duplicate of the already-numeric 'education-num'
#   - 'fnlwgt' is a Census sampling weight, not a demographic/employment
#     attribute of the individual, and is not intended as a predictor
X = df.drop(columns=["class", "education", "fnlwgt"])

numeric_features = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
categorical_features = [c for c in X.columns if c not in numeric_features]

# Missing values appear as NaN in workclass, occupation, native-country.
# Impute categoricals with a distinct "Missing" category (informative missingness
# is plausible here - e.g. missing workclass often co-occurs with missing occupation).
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ---------------------------------------------------------------------------
# 2. Train/test split (stratified, 80/20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# ---------------------------------------------------------------------------
# 3. Define distinct model families with fixed, reasonable hyperparameters
#    (no exhaustive tuning, so comparison reflects "out of the box" family
#    performance rather than best-case tuned performance).
# ---------------------------------------------------------------------------
models = {
    "LogisticRegression": LogisticRegression(
        max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=2,
        class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE
    ),
    "HistGradientBoosting": HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, random_state=RANDOM_STATE
    ),
    "KNN": KNeighborsClassifier(n_neighbors=25, n_jobs=-1),
    "GaussianNB": GaussianNB(),
}

# ---------------------------------------------------------------------------
# 4. Fit on train, evaluate on held-out test set; also run 5-fold CV
#    (on the training set) to get a stability estimate of ROC-AUC per model.
# ---------------------------------------------------------------------------
results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = {}

for name, clf in models.items():
    pipe = Pipeline(steps=[("prep", preprocessor), ("clf", clf)])

    # Cross-validated ROC-AUC on training data
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_scores[name] = scores

    # Fit on full training set, evaluate on held-out test set
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = pipe.predict(X_test)

    results[name] = {
        "test_roc_auc": roc_auc_score(y_test, proba),
        "test_pr_auc": average_precision_score(y_test, proba),
        "test_accuracy": accuracy_score(y_test, pred),
        "test_f1": f1_score(y_test, pred),
        "cv_roc_auc_mean": scores.mean(),
        "cv_roc_auc_std": scores.std(),
    }

# ---------------------------------------------------------------------------
# 5. Summarize: range across families, and a paired test between best & worst
#    (paired across the 5 CV folds) to check whether the gap is "meaningful"
#    beyond fold-to-fold noise.
# ---------------------------------------------------------------------------
ranked = sorted(results.items(), key=lambda kv: kv[1]["test_roc_auc"], reverse=True)
best_name, best_res = ranked[0]
worst_name, worst_res = ranked[-1]

roc_auc_range = best_res["test_roc_auc"] - worst_res["test_roc_auc"]

paired_t, paired_p = stats.ttest_rel(cv_scores[best_name], cv_scores[worst_name])

print("=== Test-set performance by model family ===")
for name, res in ranked:
    print(f"{name:22s} ROC-AUC={res['test_roc_auc']:.4f}  PR-AUC={res['test_pr_auc']:.4f}  "
          f"Acc={res['test_accuracy']:.4f}  F1={res['test_f1']:.4f}  "
          f"CV ROC-AUC={res['cv_roc_auc_mean']:.4f}+/-{res['cv_roc_auc_std']:.4f}")

print(f"\nBest: {best_name} ({best_res['test_roc_auc']:.4f}) | "
      f"Worst: {worst_name} ({worst_res['test_roc_auc']:.4f})")
print(f"ROC-AUC range (best - worst) = {roc_auc_range:.4f}")
print(f"Paired t-test on 5-fold CV ROC-AUC, {best_name} vs {worst_name}: "
      f"t={paired_t:.3f}, p={paired_p:.5f}")

# ---------------------------------------------------------------------------
# 6. Write result.json
# ---------------------------------------------------------------------------
meaningful = roc_auc_range > 0.01 and paired_p < 0.05

summary = (
    f"Model family matters: held-out ROC-AUC ranged from {worst_res['test_roc_auc']:.3f} "
    f"({worst_name}) to {best_res['test_roc_auc']:.3f} ({best_name}), a gap of "
    f"{roc_auc_range:.3f} that is statistically significant across 5-fold CV "
    f"(paired t-test p={paired_p:.4f}), though all families clearly beat chance "
    f"and the top few (tree ensembles vs. logistic regression) differ only modestly."
    if meaningful else
    f"Model family has little effect: held-out ROC-AUC ranged only from "
    f"{worst_res['test_roc_auc']:.3f} ({worst_name}) to {best_res['test_roc_auc']:.3f} "
    f"({best_name}), a gap of {roc_auc_range:.3f} that is not clearly distinguishable "
    f"from fold-to-fold noise (paired t-test p={paired_p:.4f})."
)

result = {
    "hypothesis_id": "H1",
    "summary": summary,
    "primary_metric_name": f"Test ROC-AUC range across model families ({best_name} - {worst_name})",
    "primary_metric_value": round(roc_auc_range, 4),
    "direction": f"{best_name} > {worst_name}" if roc_auc_range > 0 else "no clear ordering",
    "methodological_choices": (
        "Target binarized as class=='>50K'. Dropped 'fnlwgt' (Census sampling weight, "
        "not a subject attribute) and 'education' (redundant string duplicate of numeric "
        "'education-num'). Missing categorical values (workclass, occupation, "
        "native-country) imputed as an explicit 'Missing' category rather than dropped "
        "or mode-imputed, since missingness may be informative. Numeric features "
        "median-imputed (none actually missing) and standardized; categoricals one-hot "
        "encoded. Identical preprocessing pipeline used for every model family for a fair "
        "comparison (no per-model feature engineering, e.g. no ordinal/target encoding "
        "for tree models). Stratified 80/20 train/test split, random_state=42. Five model "
        "families compared with fixed, reasonable (not exhaustively tuned) "
        "hyperparameters: Logistic Regression (class_weight='balanced'), Random Forest "
        "(300 trees, class_weight='balanced'), HistGradientBoosting (300 iterations), "
        "KNN (k=25), and Gaussian Naive Bayes. Primary metric is ROC-AUC on the held-out "
        "test set (robust to the ~24%/76% class imbalance; also reported PR-AUC, accuracy, "
        "F1). Stability/significance assessed via 5-fold stratified CV ROC-AUC on the "
        "training set and a paired t-test (across folds) between the best- and "
        "worst-performing families; 'meaningful' defined as ROC-AUC range > 0.01 and "
        "paired-CV p < 0.05."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
