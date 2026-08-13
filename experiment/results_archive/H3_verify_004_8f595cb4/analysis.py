"""
H3: Which features are most important for predicting income (`class`) in the
UCI/OpenML Adult Census Income dataset?

Methodology summary
--------------------
- Drop `fnlwgt` (Census sampling weight; not a causal/individual attribute).
- Drop `education` (redundant 1:1 encoding of `education-num`).
- Impute missing categoricals (`workclass`, `occupation`, `native-country`)
  with an explicit "Missing" category rather than dropping rows (~7% of rows
  have at least one missing value).
- One-hot encode categoricals; numeric features passed through.
- Primary model: RandomForestClassifier (handles non-linearity/interactions
  well, robust default for tabular mixed-type data). A LogisticRegression
  baseline is fit for context (ROC-AUC comparison), not as the importance
  source.
- Feature importance: permutation importance on a held-out test set (not
  impurity-based), which is unbiased with respect to feature cardinality
  (a known problem with RF impurity importance) and reflects true predictive
  contribution to a scoring metric (ROC-AUC).
- Importance is aggregated to the *original* feature level (summing/grouping
  one-hot columns back to their source column) so "occupation" is judged as
  one feature rather than fragmented into 14 dummy columns.
- Stability check: repeated permutation importance across 10 different
  random splits/seeds (fresh train/test split + fresh model fit each time),
  reporting mean +/- std rank and importance for the top features, plus a
  fully independent re-test on a split never used for the primary analysis.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

y = (df["class"].str.strip() == ">50K").astype(int)

drop_cols = ["class", "fnlwgt", "education"]  # fnlwgt = sampling weight; education dup of education-num
X = df.drop(columns=drop_cols).copy()

cat_cols = X.select_dtypes(include="object").columns.tolist()
# object columns backed by pandas "str" dtype may show up differently; catch string dtype too
cat_cols = [c for c in X.columns if X[c].dtype == object or str(X[c].dtype) == "str"]
num_cols = [c for c in X.columns if c not in cat_cols]

for c in cat_cols:
    X[c] = X[c].astype("object").where(X[c].notna(), "Missing")

print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)
print("Positive class rate (>50K):", y.mean().round(4))

# ---------------------------------------------------------------------------
# 2. Preprocessing pipeline
# ---------------------------------------------------------------------------
def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def get_feature_group_map(preprocessor):
    """Map each expanded (one-hot) column name back to its source feature."""
    ohe = preprocessor.named_transformers_["cat"]
    cat_expanded = ohe.get_feature_names_out(cat_cols)
    all_expanded = list(num_cols) + list(cat_expanded)
    group = []
    for name in all_expanded:
        if name in num_cols:
            group.append(name)
        else:
            for c in cat_cols:
                if name.startswith(c + "_"):
                    group.append(c)
                    break
    return all_expanded, group


def aggregate_importance(importances_per_column, group_labels):
    s = pd.Series(importances_per_column, index=group_labels)
    return s.groupby(level=0).sum().sort_values(ascending=False)


# ---------------------------------------------------------------------------
# 3. Primary train/test split + model fit
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=RANDOM_STATE
)

pre = make_preprocessor()
rf = RandomForestClassifier(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
)
rf_pipe = Pipeline([("pre", pre), ("clf", rf)])
rf_pipe.fit(X_train, y_train)

rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
rf_auc = roc_auc_score(y_test, rf_proba)
print(f"\nRandomForest test ROC-AUC: {rf_auc:.4f}")

# Logistic regression baseline for context
logreg = LogisticRegression(max_iter=2000, class_weight="balanced")
logreg_pipe = Pipeline([("pre", make_preprocessor()), ("clf", logreg)])
logreg_pipe.fit(X_train, y_train)
logreg_proba = logreg_pipe.predict_proba(X_test)[:, 1]
logreg_auc = roc_auc_score(y_test, logreg_proba)
print(f"LogisticRegression test ROC-AUC: {logreg_auc:.4f}")

# ---------------------------------------------------------------------------
# 4. Permutation importance (primary importance method) on held-out test set
# ---------------------------------------------------------------------------
fitted_pre = rf_pipe.named_steps["pre"]
X_test_transformed = fitted_pre.transform(X_test)
if hasattr(X_test_transformed, "toarray"):
    X_test_transformed = X_test_transformed.toarray()

expanded_names, group_labels = get_feature_group_map(fitted_pre)

perm = permutation_importance(
    rf_pipe.named_steps["clf"], X_test_transformed, y_test,
    scoring="roc_auc", n_repeats=15, random_state=RANDOM_STATE, n_jobs=-1
)

primary_importance = aggregate_importance(perm.importances_mean, group_labels)
print("\nPrimary permutation importance (grouped by original feature), ROC-AUC drop:")
print(primary_importance)

top_feature = primary_importance.index[0]
top_value = float(primary_importance.iloc[0])
print(f"\nTop feature: {top_feature} (importance={top_value:.4f})")

# ---------------------------------------------------------------------------
# 5. Stability check: repeat across 10 fresh random splits/seeds
# ---------------------------------------------------------------------------
print("\n--- Stability check: 10 repeated fresh train/test splits ---")
seeds = list(range(1, 11))
rank1_counts = {}
importance_records = []

for seed in seeds:
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )
    pre_s = make_preprocessor()
    rf_s = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, n_jobs=-1,
        random_state=seed, class_weight="balanced"
    )
    pipe_s = Pipeline([("pre", pre_s), ("clf", rf_s)])
    pipe_s.fit(Xtr, ytr)

    fitted_pre_s = pipe_s.named_steps["pre"]
    Xte_t = fitted_pre_s.transform(Xte)
    if hasattr(Xte_t, "toarray"):
        Xte_t = Xte_t.toarray()
    _, group_labels_s = get_feature_group_map(fitted_pre_s)

    perm_s = permutation_importance(
        pipe_s.named_steps["clf"], Xte_t, yte,
        scoring="roc_auc", n_repeats=8, random_state=seed, n_jobs=-1
    )
    imp_s = aggregate_importance(perm_s.importances_mean, group_labels_s)
    importance_records.append(imp_s)

    top = imp_s.index[0]
    rank1_counts[top] = rank1_counts.get(top, 0) + 1
    print(f"seed={seed:2d}  top feature={top:15s}  value={imp_s.iloc[0]:.4f}  auc={roc_auc_score(yte, pipe_s.predict_proba(Xte)[:,1]):.4f}")

imp_df = pd.DataFrame(importance_records).fillna(0)
summary = imp_df.agg(["mean", "std"]).T.sort_values("mean", ascending=False)
print("\nMean +/- std permutation importance across 10 seeds:")
print(summary)

print("\nTimes each feature ranked #1 across 10 seeds:", rank1_counts)

stable_top_feature = summary.index[0]
stable_top_mean = float(summary.loc[stable_top_feature, "mean"])
stable_top_std = float(summary.loc[stable_top_feature, "std"])

# ---------------------------------------------------------------------------
# 6. Independent held-out re-test split (never touched above)
# ---------------------------------------------------------------------------
print("\n--- Independent re-test on a fresh, previously-unused split ---")
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=999
)
pre2 = make_preprocessor()
rf2 = RandomForestClassifier(
    n_estimators=400, min_samples_leaf=2, n_jobs=-1,
    random_state=999, class_weight="balanced"
)
pipe2 = Pipeline([("pre", pre2), ("clf", rf2)])
pipe2.fit(X_train2, y_train2)

fitted_pre2 = pipe2.named_steps["pre"]
X_test2_t = fitted_pre2.transform(X_test2)
if hasattr(X_test2_t, "toarray"):
    X_test2_t = X_test2_t.toarray()
_, group_labels2 = get_feature_group_map(fitted_pre2)

perm2 = permutation_importance(
    pipe2.named_steps["clf"], X_test2_t, y_test2,
    scoring="roc_auc", n_repeats=15, random_state=999, n_jobs=-1
)
imp2 = aggregate_importance(perm2.importances_mean, group_labels2)
print(imp2)
print(f"\nRe-test top feature: {imp2.index[0]} (importance={imp2.iloc[0]:.4f})")
print(f"Re-test ROC-AUC: {roc_auc_score(y_test2, pipe2.predict_proba(X_test2)[:,1]):.4f}")

# ---------------------------------------------------------------------------
# 7. Final report
# ---------------------------------------------------------------------------
print("\n=== FINAL SUMMARY ===")
print(f"Primary-split top feature: {top_feature} ({top_value:.4f})")
print(f"10-seed mean-importance top feature: {stable_top_feature} ({stable_top_mean:.4f} +/- {stable_top_std:.4f})")
print(f"Independent re-test top feature: {imp2.index[0]} ({imp2.iloc[0]:.4f})")
print(f"Rank-1 consistency across 10 seeds: {rank1_counts}")
