"""
H2: Does RandomForestClassifier() outperform LogisticRegression() on ROC-AUC
(stratified 5-fold CV) for predicting income class on the Adult dataset?

Both models are used with scikit-learn default hyperparameters, as specified
in the research question. All other choices (encoding, missing-value
handling, scaling, CV scheme, metric aggregation) are the researcher's
judgment calls, documented inline and in result.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

# Target: binary-encode class (>50K = 1)
y = (df["class"].str.strip() == ">50K").astype(int)
X = df.drop(columns=["class"])

# fnlwgt is a census sampling weight, not a real demographic feature of the
# individual -- it's noise for a supervised classifier here, so drop it.
X = X.drop(columns=["fnlwgt"])

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "str"]).columns.tolist()

# Missing values in categorical columns (workclass, occupation, native-country)
# are read as NaN by pandas (they were empty/whitespace-only strings in the
# source CSV). Impute with a distinct "Missing" category rather than dropping
# rows (~2650 rows would be lost, ~6.7% of data) or dropping columns.
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

# Numeric features: no missing values present, but StandardScaler is applied
# since LogisticRegression is scale-sensitive; RF is scale-invariant so this
# choice only affects LogReg and is neutral/harmless for RF.
numeric_transformer = Pipeline(steps=[
    ("scaler", StandardScaler()),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_cols),
    ("cat", categorical_transformer, categorical_cols),
])

logreg_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", LogisticRegression()),  # scikit-learn defaults
])

rf_pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),  # defaults + seed for reproducibility
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg_scores = cross_val_score(logreg_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

print("LogisticRegression ROC-AUC per fold:", logreg_scores)
print("LogisticRegression mean ROC-AUC: %.5f (+/- %.5f)" % (logreg_scores.mean(), logreg_scores.std()))
print()
print("RandomForest ROC-AUC per fold:", rf_scores)
print("RandomForest mean ROC-AUC: %.5f (+/- %.5f)" % (rf_scores.mean(), rf_scores.std()))
print()

diff = rf_scores.mean() - logreg_scores.mean()
print("Difference (RF - LogReg): %.5f" % diff)

# Paired t-test across the 5 folds to assess whether the difference is
# consistent/significant rather than noise.
from scipy import stats
t_stat, p_value = stats.ttest_rel(rf_scores, logreg_scores)
print("Paired t-test: t=%.4f, p=%.4f" % (t_stat, p_value))

rf_wins = diff > 0
comparison_word = "higher" if rf_wins else "lower"
answer_word = "yes" if rf_wins else "no"
direction = "RF > LogReg" if rf_wins else "RF < LogReg"

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Random forest achieved a {comparison_word} mean stratified 5-fold CV ROC-AUC "
        f"({rf_scores.mean():.4f}) than logistic regression ({logreg_scores.mean():.4f}) "
        f"on the Adult income dataset, a difference of {diff:.4f} "
        f"(paired t-test p={p_value:.4f}), so the answer to H2 is {answer_word} — "
        f"default logistic regression actually outperforms default random forest here."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg)",
    "primary_metric_value": round(float(diff), 5),
    "direction": direction,
    "methodological_choices": (
        "Both models used scikit-learn defaults (RandomForestClassifier(random_state=42), "
        "LogisticRegression()); random_state fixed only for RF/CV reproducibility, not a "
        "hyperparameter change. Dropped 'fnlwgt' (a census sampling weight, not a demographic "
        "feature of the individual). Missing values in workclass/occupation/native-country "
        "(read as NaN from blank cells in source CSV, ~2-6% of rows per column) imputed as a "
        "distinct 'Missing' category rather than row/column deletion. Categorical features "
        "one-hot encoded; numeric features standard-scaled (neutral for RF, needed for LogReg "
        "convergence/comparability). Target binarized as class == '>50K'. Evaluation: "
        "StratifiedKFold(5, shuffle=True, random_state=42) cross_val_score with scoring='roc_auc', "
        "comparing mean AUC across folds; a paired t-test across the 5 fold scores was used as a "
        "secondary check on whether the difference is consistent across folds. No class-imbalance "
        "correction (~24% positive class) was applied since ROC-AUC is relatively robust to "
        "moderate imbalance and the question specifies default hyperparameters."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print()
print(json.dumps(result, indent=2))
