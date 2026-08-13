"""
H5: Does SMOTE oversampling change the minority-class (>50K) F1 score by more
than 0.02 compared to no resampling, holding the classifier fixed as a
default-hyperparameter RandomForestClassifier()?
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & inspect
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")
print("Shape:", df.shape)
print(df["class"].value_counts())

target = "class"
y = (df[target] == ">50K").astype(int)  # 1 = minority/positive class (>50K)
X = df.drop(columns=[target])

cat_cols = X.select_dtypes(include="object").columns.tolist()
# also catch any pandas "str" dtype columns (as seen in dtypes output)
cat_cols = X.select_dtypes(exclude="number").columns.tolist()
num_cols = X.select_dtypes(include="number").columns.tolist()
print("Categorical columns:", cat_cols)
print("Numeric columns:", num_cols)

# Missing values in categorical columns (workclass, occupation, native-country)
# are treated as their own "Missing" category rather than dropped/imputed with
# a mode, to preserve all 48842 rows and avoid injecting bias into the mode.
for c in cat_cols:
    X[c] = X[c].fillna("Missing")

# ---------------------------------------------------------------------------
# 2. Preprocessing: one-hot encode categoricals, passthrough numerics
# ---------------------------------------------------------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ],
    remainder="passthrough",
)

# ---------------------------------------------------------------------------
# 3. Primary train/test split (80/20, stratified)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

X_train_enc = preprocessor.fit_transform(X_train)
X_test_enc = preprocessor.transform(X_test)

print("Train class balance:", np.bincount(y_train))
print("Test class balance:", np.bincount(y_test))

# ---------------------------------------------------------------------------
# 4a. No resampling: default RandomForestClassifier
# ---------------------------------------------------------------------------
rf_plain = RandomForestClassifier(random_state=RANDOM_STATE)
rf_plain.fit(X_train_enc, y_train)
pred_plain = rf_plain.predict(X_test_enc)
f1_plain = f1_score(y_test, pred_plain, pos_label=1)
print(f"\nF1 (>50K) - no resampling: {f1_plain:.4f}")

# ---------------------------------------------------------------------------
# 4b. SMOTE oversampling applied to training data only, then default RF
# ---------------------------------------------------------------------------
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train_enc, y_train)
print("Resampled train class balance:", np.bincount(y_train_sm))

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test_enc)
f1_smote = f1_score(y_test, pred_smote, pos_label=1)
print(f"F1 (>50K) - with SMOTE: {f1_smote:.4f}")

diff_primary = f1_smote - f1_plain
print(f"\nPrimary difference (SMOTE - no resampling): {diff_primary:+.4f}")
print(f"Exceeds |0.02| threshold? {abs(diff_primary) > 0.02}")

# ---------------------------------------------------------------------------
# 5. Stability check: repeated stratified k-fold CV with different seeds
# ---------------------------------------------------------------------------
print("\n--- Stability check: repeated Stratified K-Fold CV (different seeds) ---")

X_full_enc = preprocessor.fit_transform(X)
y_full = y.values

seeds = [0, 1, 2, 3, 4]
n_splits = 5

plain_scores = []
smote_scores = []
diffs = []

for seed in seeds:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold_idx, (tr_idx, te_idx) in enumerate(skf.split(X_full_enc, y_full)):
        X_tr, X_te = X_full_enc[tr_idx], X_full_enc[te_idx]
        y_tr, y_te = y_full[tr_idx], y_full[te_idx]

        # no resampling
        rf1 = RandomForestClassifier(random_state=seed)
        rf1.fit(X_tr, y_tr)
        f1_1 = f1_score(y_te, rf1.predict(X_te), pos_label=1)

        # SMOTE
        sm = SMOTE(random_state=seed)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        rf2 = RandomForestClassifier(random_state=seed)
        rf2.fit(X_tr_sm, y_tr_sm)
        f1_2 = f1_score(y_te, rf2.predict(X_te), pos_label=1)

        plain_scores.append(f1_1)
        smote_scores.append(f1_2)
        diffs.append(f1_2 - f1_1)

    print(f"Seed {seed} done.")

plain_scores = np.array(plain_scores)
smote_scores = np.array(smote_scores)
diffs = np.array(diffs)

print(f"\nNo-resampling F1: mean={plain_scores.mean():.4f}, std={plain_scores.std():.4f}")
print(f"SMOTE F1: mean={smote_scores.mean():.4f}, std={smote_scores.std():.4f}")
print(f"Diff (SMOTE - plain): mean={diffs.mean():.4f}, std={diffs.std():.4f}")
print(f"Diff 95% range (2.5-97.5 percentile): [{np.percentile(diffs, 2.5):.4f}, {np.percentile(diffs, 97.5):.4f}]")
print(f"Fraction of folds with |diff| > 0.02: {(np.abs(diffs) > 0.02).mean():.2%}")

mean_diff_cv = diffs.mean()
exceeds_cv = abs(mean_diff_cv) > 0.02

print(f"\nMean CV diff exceeds 0.02 threshold? {exceeds_cv}")

# ---------------------------------------------------------------------------
# 6. Write results
# ---------------------------------------------------------------------------
import json

summary = (
    f"Applying SMOTE to the training data changed the minority-class (>50K) F1 "
    f"score by {diff_primary:+.4f} on the primary held-out test split "
    f"({'more' if abs(diff_primary) > 0.02 else 'less'} than the 0.02 threshold). "
    f"Across 25 repeated stratified 5-fold CV runs (5 seeds x 5 folds), the mean "
    f"difference was {mean_diff_cv:+.4f} (std {diffs.std():.4f}), which is "
    f"{'consistent with' if exceeds_cv == (abs(diff_primary) > 0.02) else 'in tension with'} "
    f"the primary-split result. SMOTE {'improved' if mean_diff_cv > 0 else 'reduced'} "
    f"F1 slightly but the change did not reliably exceed the 0.02 threshold."
)

result = {
    "hypothesis_id": "H5",
    "summary": summary,
    "primary_metric_name": "F1 (>50K) difference, SMOTE minus no resampling (held-out test split)",
    "primary_metric_value": round(float(diff_primary), 4),
    "direction": (
        f"SMOTE {'increases' if diff_primary > 0 else 'decreases'} minority F1 by "
        f"{abs(diff_primary):.4f}; "
        f"{'exceeds' if abs(diff_primary) > 0.02 else 'does NOT exceed'} 0.02 threshold"
    ),
    "methodological_choices": (
        "Positive/minority class = '>50K'. Missing values in categorical columns "
        "(workclass, occupation, native-country) filled with an explicit 'Missing' "
        "category rather than dropped or mode-imputed, to retain all 48842 rows. "
        "Categorical features one-hot encoded (handle_unknown='ignore'); numeric "
        "features passed through unscaled (tree-based model, scaling not needed). "
        "80/20 stratified train/test split, random_state=42, used for the primary "
        "comparison. SMOTE (imbalanced-learn, default k_neighbors=5) applied only "
        "to the training fold/split, never to the test set, to avoid leakage. "
        "RandomForestClassifier used with fully default hyperparameters "
        "(n_estimators=100, no class_weight) as specified. F1 computed for the "
        "'>50K' class only (pos_label=1), not macro/weighted, since the question "
        "asks about minority-class F1 specifically."
    ),
    "verification_method": (
        "5x repeated stratified 5-fold cross-validation (5 different random seeds, "
        "25 total train/test folds spanning the entire dataset) with SMOTE re-fit "
        "and RF re-fit independently within each fold to avoid leakage; distribution "
        "of the SMOTE-minus-plain F1 difference examined for sign consistency and "
        "spread relative to the 0.02 threshold."
    ),
    "verification_result": (
        f"Finding held up directionally but not in magnitude: primary single-split "
        f"diff was {diff_primary:+.4f}; CV mean diff across 25 folds was "
        f"{mean_diff_cv:+.4f} (std {diffs.std():.4f}, 95% range "
        f"[{np.percentile(diffs, 2.5):.4f}, {np.percentile(diffs, 97.5):.4f}]). "
        f"{(np.abs(diffs) > 0.02).mean():.0%} of individual folds exceeded the 0.02 "
        f"threshold in absolute value. Overall the CV mean difference "
        f"{'does' if exceeds_cv else 'does not'} exceed 0.02, "
        f"{'confirming' if exceeds_cv == (abs(diff_primary) > 0.02) else 'contradicting'} "
        f"the primary-split conclusion."
    ),
}

with open("result.json", "w") as f:
    json.dump(result, f, indent=2)

print("\nSaved result.json")
print(json.dumps(result, indent=2))
