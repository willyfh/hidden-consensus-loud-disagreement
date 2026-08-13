"""
H2: Does RandomForestClassifier (sklearn defaults) achieve higher stratified
5-fold CV ROC-AUC than LogisticRegression (sklearn defaults) on the Adult
Income dataset?
"""
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

target = "class"
y = (df[target] == ">50K").astype(int)
X = df.drop(columns=[target])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

# Missing values appear as NaN in categorical columns (workclass, occupation,
# native-country). Impute categoricals with a constant "Missing" category
# (missingness itself can be informative, e.g. never-worked -> NaN occupation)
# rather than dropping ~2800-2900 rows, to preserve sample size.
numeric_pipe = Pipeline([
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

# Logistic regression needs scaled numeric features and one-hot categoricals;
# it also needs a higher max_iter than the sklearn default (100) to converge
# on this one-hot-expanded feature space, so that's bumped while leaving
# every other hyperparameter at sklearn defaults per the research question.
logreg_pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
])

# Random forest is scale-invariant, so it reuses the same encoded feature
# matrix (one-hot + passthrough-scaled numerics) for a fair apples-to-apples
# comparison on identical input features; scaling numerics does not affect
# an RF's splits.
rf_pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg_scores = cross_val_score(logreg_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

print("Logistic Regression ROC-AUC per fold:", logreg_scores)
print("Logistic Regression mean ROC-AUC: %.5f (+/- %.5f)" % (logreg_scores.mean(), logreg_scores.std()))
print("Random Forest ROC-AUC per fold:", rf_scores)
print("Random Forest mean ROC-AUC: %.5f (+/- %.5f)" % (rf_scores.mean(), rf_scores.std()))

diff = rf_scores.mean() - logreg_scores.mean()
print("Difference (RF - LogReg): %.5f" % diff)

# Paired t-test across the 5 folds to gauge whether the difference is
# consistent/significant rather than noise from a single split.
from scipy import stats
t_stat, p_value = stats.ttest_rel(rf_scores, logreg_scores)
print("Paired t-test: t=%.4f, p=%.5f" % (t_stat, p_value))

direction = "RF > LogReg" if diff > 0 else ("RF < LogReg" if diff < 0 else "RF == LogReg")

result = {
    "hypothesis_id": "H2",
    "summary": (
        f"Random forest achieved a mean stratified 5-fold CV ROC-AUC of {rf_scores.mean():.4f} "
        f"versus {logreg_scores.mean():.4f} for logistic regression on the Adult Income dataset "
        f"({'higher' if diff > 0 else 'lower'} by {abs(diff):.4f}), so the random forest "
        f"{'does' if diff > 0 else 'does not'} outperform logistic regression on this metric."
    ),
    "primary_metric_name": "ROC-AUC difference (RF - LogReg), mean over 5 folds",
    "primary_metric_value": float(diff),
    "direction": direction,
    "methodological_choices": (
        "Target encoded as binary (1 = '>50K'). Missing values in workclass/occupation/"
        "native-country (originally NaN, ~1.7-5.7% of rows) were imputed as a constant "
        "'Missing' category rather than dropped, to preserve full sample size of 48842 rows. "
        "Numeric features (age, fnlwgt, education-num, capital-gain, capital-loss, "
        "hours-per-week) were standardized (StandardScaler); categorical features "
        "(workclass, education, marital-status, occupation, relationship, race, sex, "
        "native-country) were one-hot encoded (handle_unknown='ignore'). Identical "
        "preprocessed feature matrix used for both models for a fair comparison, even "
        "though RF does not require scaling and could in principle use raw/ordinal-encoded "
        "features. Both classifiers used scikit-learn default hyperparameters as specified "
        "by the research question, except LogisticRegression max_iter was raised from 100 to "
        "1000 solely to achieve convergence on the expanded one-hot feature space (this does "
        "not change the model's default regularization or solver). Evaluation used "
        "StratifiedKFold(n_splits=5, shuffle=True, random_state=42) and scoring='roc_auc' "
        "(binary ROC-AUC using predicted probability of class '>50K'). A paired t-test across "
        "the 5 fold scores was used to assess whether the RF-LogReg difference is consistent "
        "across folds. education-num was kept as numeric (it is an ordinal encoding of the "
        "categorical education column) and education was also kept as one-hot categorical, "
        "which is redundant but harmless for both models. random_state=42 fixed throughout "
        "for reproducibility."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print(json.dumps(result, indent=2))
