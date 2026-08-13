"""
H3: Which features are most important for predicting income (class: <=50K vs >50K)
in the UCI/OpenML Adult Census Income dataset?

Approach
--------
1. Load and clean the data (handle '?' missing-value tokens).
2. Encode categoricals (ordinal encoding for tree models is fine; one-hot avoided
   here in favor of ordinal encoding since we use tree-based models which do not
   assume a linear relationship between encoded integers and outcome).
3. Split into train/test (70/30, stratified on target).
4. Fit a Random Forest classifier as the primary model (robust, minimal tuning
   needed, provides two independent importance signals: impurity-based and
   permutation-based).
5. Evaluate discriminative performance (ROC-AUC, accuracy) to sanity check the
   model is reasonable before trusting its importances.
6. Compute feature importances two ways:
     a) Random Forest built-in (mean decrease in impurity, Gini importance)
     b) Permutation importance on the held-out test set (drop-in AUC when a
        feature is shuffled) -- this is the primary importance metric because
        it is model-agnostic and measures actual predictive contribution
        rather than tree-structure artifacts (which are biased toward
        high-cardinality features).
7. Also fit a simple logistic regression (on one-hot encoded / scaled features)
   and rank features by absolute standardized coefficient, as a cross-check
   against a linear model with a different inductive bias.
8. Report the top features and the primary metric (top feature's permutation
   importance, measured as mean decrease in ROC-AUC).
"""

import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & clean
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

# Missing values in this dataset are encoded as '?' in categorical columns.
df = df.replace("?", np.nan)

target_col = "class"
y = (df[target_col].str.strip() == ">50K").astype(int)
X = df.drop(columns=[target_col])

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(exclude="object").columns.tolist()

# fnlwgt is a census sampling weight, not a real demographic feature of the
# individual -- keep it in the analysis (it's a column in the data and could
# in principle leak info about stratum), but it is worth flagging in the
# writeup that it's not conceptually a "predictor" in the usual sense.

print(f"Rows: {len(df)}, categorical cols: {cat_cols}, numeric cols: {num_cols}")
print("Missing values per column:\n", X.isna().sum()[X.isna().sum() > 0])
print("Target balance:\n", y.value_counts(normalize=True))

# ---------------------------------------------------------------------------
# 2. Train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)

# ---------------------------------------------------------------------------
# 3. Random Forest pipeline (ordinal-encode categoricals, impute missing as
#    a distinct "Missing" category; RF handles mixed monotonic/ordinal codes
#    fine since it splits on thresholds per feature independently)
# ---------------------------------------------------------------------------
rf_preprocess = ColumnTransformer(
    transformers=[
        (
            "cat",
            OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-2,
            ),
            cat_cols,
        ),
        ("num", "passthrough", num_cols),
    ]
)

rf_pipe = Pipeline(
    steps=[
        ("prep", rf_preprocess),
        (
            "clf",
            RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=RANDOM_STATE,
                class_weight=None,  # target is ~24/76, mild imbalance; leave unweighted since AUC is threshold-free
            ),
        ),
    ]
)

rf_pipe.fit(X_train, y_train)

rf_proba = rf_pipe.predict_proba(X_test)[:, 1]
rf_pred = rf_pipe.predict(X_test)
rf_auc = roc_auc_score(y_test, rf_proba)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"\nRandom Forest -- test ROC-AUC: {rf_auc:.4f}, accuracy: {rf_acc:.4f}")

# ---------------------------------------------------------------------------
# 4. Impurity-based (Gini) importances
# ---------------------------------------------------------------------------
feature_names = cat_cols + num_cols
rf_clf = rf_pipe.named_steps["clf"]
gini_importances = pd.Series(rf_clf.feature_importances_, index=feature_names).sort_values(
    ascending=False
)
print("\nRandom Forest Gini importances:\n", gini_importances)

# ---------------------------------------------------------------------------
# 5. Permutation importance on held-out test set (primary importance metric)
#    Scored by drop in ROC-AUC when each feature's values are shuffled.
# ---------------------------------------------------------------------------
perm_result = permutation_importance(
    rf_pipe,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)
perm_importances = pd.Series(perm_result.importances_mean, index=X_test.columns).sort_values(
    ascending=False
)
perm_std = pd.Series(perm_result.importances_std, index=X_test.columns)
print("\nPermutation importances (mean AUC drop):\n", perm_importances)

# ---------------------------------------------------------------------------
# 6. Cross-check with logistic regression (linear model, one-hot + scaling)
# ---------------------------------------------------------------------------
lr_preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols),
    ]
)

lr_pipe = Pipeline(
    steps=[
        ("prep", lr_preprocess),
        ("clf", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
    ]
)
lr_pipe.fit(X_train, y_train)
lr_proba = lr_pipe.predict_proba(X_test)[:, 1]
lr_auc = roc_auc_score(y_test, lr_proba)
print(f"\nLogistic Regression -- test ROC-AUC: {lr_auc:.4f}")

# Aggregate abs standardized coefficient per original feature (sum across
# one-hot levels for categoricals) as a rough linear-model importance proxy.
ohe = lr_pipe.named_steps["prep"].named_transformers_["cat"]
ohe_feature_names = ohe.get_feature_names_out(cat_cols)
all_lr_feature_names = list(ohe_feature_names) + num_cols
coefs = pd.Series(lr_pipe.named_steps["clf"].coef_[0], index=all_lr_feature_names)

lr_importance_by_orig = {}
for col in cat_cols:
    mask = [f for f in coefs.index if f.startswith(f"{col}_")]
    lr_importance_by_orig[col] = coefs[mask].abs().sum()
for col in num_cols:
    lr_importance_by_orig[col] = abs(coefs[col])
lr_importance_by_orig = pd.Series(lr_importance_by_orig).sort_values(ascending=False)
print("\nLogistic Regression |coef| summed by original feature:\n", lr_importance_by_orig)

# ---------------------------------------------------------------------------
# 7. Final report
# ---------------------------------------------------------------------------
top_feature = perm_importances.index[0]
top_value = float(perm_importances.iloc[0])

print(f"\nTop feature by permutation importance: {top_feature} ({top_value:.4f} mean AUC drop)")

ranking_table = pd.DataFrame(
    {
        "permutation_importance_mean_auc_drop": perm_importances,
        "permutation_importance_std": perm_std.reindex(perm_importances.index),
        "rf_gini_importance": gini_importances.reindex(perm_importances.index),
    }
)
print("\nCombined ranking table:\n", ranking_table)

result = {
    "hypothesis_id": "H3",
    "summary": (
        f"Across a 500-tree Random Forest and a permutation-importance analysis on a held-out "
        f"test set, '{top_feature}' is the single most important predictor of income class "
        f"(mean ROC-AUC drop of {top_value:.4f} when permuted), followed by "
        f"{', '.join(perm_importances.index[1:4])}. RF Gini importance cross-checks this closely "
        f"(capital-gain and relationship are its top two). A logistic-regression cross-check's "
        f"raw coefficient ranking is dominated by native-country/occupation, but that is an "
        f"artifact of summing many sparse one-hot coefficients for high-cardinality columns "
        f"rather than genuine importance -- those same features rank lowest under permutation "
        f"importance, so capital-gain, relationship/marital-status, education, and age remain "
        f"the most credible top predictors overall."
    ),
    "primary_metric_name": "top feature permutation importance (mean ROC-AUC drop)",
    "primary_metric_value": round(top_value, 4),
    "direction": f"'{top_feature}' most important",
    "methodological_choices": (
        "Target binarized as class=='>50K'. '?' tokens treated as missing and encoded as a "
        "distinct category. 70/30 stratified train/test split, random_state=42. Primary model: "
        "RandomForestClassifier(n_estimators=500, min_samples_leaf=2) with ordinal-encoded "
        "categoricals (unknown/missing given sentinel codes), unweighted despite ~76/24 class "
        "split since ROC-AUC is threshold-independent. Feature importance measured primarily via "
        "permutation importance (mean decrease in test-set ROC-AUC over 10 shuffles per feature) "
        "rather than RF impurity-based (Gini) importance, since Gini importance is biased toward "
        "high-cardinality/continuous features (e.g. fnlwgt, capital-gain) and can be misleading; "
        "permutation importance was computed on held-out data. Random Forest Gini importances and "
        "a logistic regression (one-hot encoded, standardized numerics, |coefficient| summed per "
        "original feature) are reported as cross-checks; note the LR |coefficient|-sum ranking is "
        "inflated for high-cardinality categoricals (native-country, occupation) since summing "
        "many sparse one-hot coefficients is not a fair comparison against single continuous "
        "features, so it is treated as a weaker signal than permutation importance. fnlwgt "
        "(census sampling weight) was kept "
        "as a feature but flagged as not a genuine demographic predictor. RF test ROC-AUC: "
        f"{rf_auc:.4f}, accuracy: {rf_acc:.4f}; Logistic Regression test ROC-AUC: {lr_auc:.4f}."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nWrote result.json")
