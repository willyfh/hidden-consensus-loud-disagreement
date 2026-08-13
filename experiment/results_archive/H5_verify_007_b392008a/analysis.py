"""
H5: Does SMOTE oversampling change minority-class (>50K) F1 for a default
RandomForestClassifier, compared to no resampling, by more than 0.02?
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42
POS_LABEL = ">50K"

# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

# Categorical columns with missing values ('workclass', 'occupation',
# 'native-country') are filled with an explicit 'Missing' category rather than
# dropped, to retain the full 48842-row sample and let the model use
# "missingness" itself as a signal (it is not missing-at-random -- e.g.
# workclass and occupation are jointly missing for people out of the labor
# force).
cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("class")
num_cols = [c for c in df.columns if c not in cat_cols + ["class"]]

for c in cat_cols:
    df[c] = df[c].fillna("Missing")

X = df[num_cols + cat_cols]
y = (df["class"] == POS_LABEL).astype(int)

preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ]
)

# ---------------------------------------------------------------------------
# 2. Primary analysis: single 80/20 stratified train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

# No resampling: default RandomForestClassifier, default hyperparameters.
pipe_plain = Pipeline(
    [
        ("prep", preprocess),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)
pipe_plain.fit(X_train, y_train)
pred_plain = pipe_plain.predict(X_test)
f1_plain = f1_score(y_test, pred_plain, pos_label=1)

# With SMOTE: oversampling applied to the training folds only (fit inside the
# pipeline so the test set is never touched by SMOTE), default SMOTE params
# (k_neighbors=5), same default RandomForestClassifier downstream.
pipe_smote = ImbPipeline(
    [
        ("prep", preprocess),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),
    ]
)
pipe_smote.fit(X_train, y_train)
pred_smote = pipe_smote.predict(X_test)
f1_smote = f1_score(y_test, pred_smote, pos_label=1)

diff_primary = f1_smote - f1_plain

print("=== Primary single-split analysis ===")
print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Class balance in train: {y_train.mean():.4f} positive")
print(f"F1 (>50K), no resampling : {f1_plain:.4f}")
print(f"F1 (>50K), SMOTE         : {f1_smote:.4f}")
print(f"Difference (SMOTE - plain): {diff_primary:.4f}")
print()

# ---------------------------------------------------------------------------
# 3. Stability check: repeated stratified 5-fold CV, 10 repeats (50 folds
#    total), each with a different random seed, to see whether the sign and
#    rough magnitude of the difference is stable across resampling of the
#    train/test partition and across the stochastic elements of SMOTE / RF.
# ---------------------------------------------------------------------------
n_splits = 5
n_repeats = 10
rskf = RepeatedStratifiedKFold(
    n_splits=n_splits, n_repeats=n_repeats, random_state=123
)

diffs = []
f1_plain_list = []
f1_smote_list = []

for fold_idx, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    seed = fold_idx  # vary the RF/SMOTE seed alongside the CV split

    p_plain = Pipeline(
        [
            ("prep", preprocess),
            ("clf", RandomForestClassifier(random_state=seed)),
        ]
    )
    p_plain.fit(X_tr, y_tr)
    f1_p = f1_score(y_te, p_plain.predict(X_te), pos_label=1)

    p_smote = ImbPipeline(
        [
            ("prep", preprocess),
            ("smote", SMOTE(random_state=seed)),
            ("clf", RandomForestClassifier(random_state=seed)),
        ]
    )
    p_smote.fit(X_tr, y_tr)
    f1_s = f1_score(y_te, p_smote.predict(X_te), pos_label=1)

    f1_plain_list.append(f1_p)
    f1_smote_list.append(f1_s)
    diffs.append(f1_s - f1_p)

diffs = np.array(diffs)
f1_plain_arr = np.array(f1_plain_list)
f1_smote_arr = np.array(f1_smote_list)

print(f"=== Stability check: {n_repeats}x repeated {n_splits}-fold CV ({len(diffs)} folds) ===")
print(f"Mean F1 no resampling : {f1_plain_arr.mean():.4f} (sd {f1_plain_arr.std():.4f})")
print(f"Mean F1 SMOTE         : {f1_smote_arr.mean():.4f} (sd {f1_smote_arr.std():.4f})")
print(f"Mean difference (SMOTE - plain): {diffs.mean():.4f}")
print(f"SD of difference across folds  : {diffs.std():.4f}")
print(f"Min / Max difference           : {diffs.min():.4f} / {diffs.max():.4f}")

# 95% CI via normal approximation on the fold-level differences
ci_lo = diffs.mean() - 1.96 * diffs.std(ddof=1) / np.sqrt(len(diffs))
ci_hi = diffs.mean() + 1.96 * diffs.std(ddof=1) / np.sqrt(len(diffs))
print(f"Approx 95% CI of mean difference: [{ci_lo:.4f}, {ci_hi:.4f}]")

exceeds_threshold = abs(diffs.mean()) > 0.02
print(f"\n|mean diff| > 0.02 ? {exceeds_threshold}")
