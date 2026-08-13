"""
H5: Does SMOTE oversampling change the minority-class (>50K) F1 score by more
than 0.02 compared to no resampling, holding the classifier fixed as a
default-hyperparameter RandomForestClassifier()?
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")

# "?" was already read as NaN by pandas (values show as nan). Treat missing
# categoricals as their own category "Missing" rather than dropping rows,
# since ~7% of rows have at least one missing categorical and dropping them
# would needlessly shrink the (already imbalanced) minority class.
cat_cols = ["workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"]
num_cols = ["age", "fnlwgt", "education-num", "capital-gain",
            "capital-loss", "hours-per-week"]

for c in cat_cols:
    df[c] = df[c].fillna("Missing")

y = (df["class"].str.strip() == ">50K").astype(int)
X = df[cat_cols + num_cols]

print("Class balance overall:", y.value_counts(normalize=True).to_dict())

# ---------------------------------------------------------------------------
# 2. Preprocessing: one-hot encode categoricals, passthrough numerics.
#    Tree-based RF doesn't need scaling.
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols),
    ]
)

# ---------------------------------------------------------------------------
# 3. Primary analysis: single stratified 70/30 train/test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
)

def make_pipeline(use_smote: bool, random_state: int) -> ImbPipeline:
    steps = [("prep", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("clf", RandomForestClassifier(random_state=random_state)))
    return ImbPipeline(steps)

def evaluate(use_smote: bool, X_tr, y_tr, X_te, y_te, random_state=RANDOM_STATE):
    pipe = make_pipeline(use_smote, random_state)
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    return {
        "f1_minority": f1_score(y_te, preds, pos_label=1),
        "precision_minority": precision_score(y_te, preds, pos_label=1),
        "recall_minority": recall_score(y_te, preds, pos_label=1),
    }

res_no_smote = evaluate(False, X_train, y_train, X_test, y_test)
res_smote = evaluate(True, X_train, y_train, X_test, y_test)

diff = res_smote["f1_minority"] - res_no_smote["f1_minority"]

print("\n--- Primary single-split result ---")
print("No SMOTE:", res_no_smote)
print("SMOTE:   ", res_smote)
print("F1 diff (SMOTE - no SMOTE):", diff)

# ---------------------------------------------------------------------------
# 4. Stability check: repeated stratified 5-fold CV with multiple seeds
# ---------------------------------------------------------------------------
seeds = [0, 1, 2, 3, 4]
n_folds = 5

no_smote_f1s = []
smote_f1s = []
diffs = []

for seed in seeds:
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        r_no = evaluate(False, X_tr, y_tr, X_te, y_te, random_state=seed)
        r_yes = evaluate(True, X_tr, y_tr, X_te, y_te, random_state=seed)

        no_smote_f1s.append(r_no["f1_minority"])
        smote_f1s.append(r_yes["f1_minority"])
        diffs.append(r_yes["f1_minority"] - r_no["f1_minority"])

no_smote_f1s = np.array(no_smote_f1s)
smote_f1s = np.array(smote_f1s)
diffs = np.array(diffs)

print("\n--- Repeated 5-fold CV (5 seeds, 25 folds total) ---")
print(f"No-SMOTE F1: mean={no_smote_f1s.mean():.4f}, std={no_smote_f1s.std():.4f}")
print(f"SMOTE F1:    mean={smote_f1s.mean():.4f}, std={smote_f1s.std():.4f}")
print(f"Diff (SMOTE-noSMOTE): mean={diffs.mean():.4f}, std={diffs.std():.4f}, "
      f"min={diffs.min():.4f}, max={diffs.max():.4f}")

# 95% CI via normal approx on the paired differences
ci_low = diffs.mean() - 1.96 * diffs.std(ddof=1) / np.sqrt(len(diffs))
ci_high = diffs.mean() + 1.96 * diffs.std(ddof=1) / np.sqrt(len(diffs))
print(f"95% CI on mean diff: [{ci_low:.4f}, {ci_high:.4f}]")

print("\nDone.")
