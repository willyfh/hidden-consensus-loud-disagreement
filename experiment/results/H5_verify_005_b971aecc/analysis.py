"""
H5: Does SMOTE oversampling change minority-class (>50K) F1 for a default
RandomForestClassifier(), compared to no resampling?

Methodology
-----------
- Load adult_income.csv (48842 rows). Target = `class` (<=50K / >50K), a 76/24 imbalance.
- Missing values (encoded as NaN, originally '?') in workclass/occupation/native-country
  are filled with the literal category "Missing" rather than dropped or mode-imputed,
  to preserve all rows and keep "missingness" as potential signal.
- Categorical columns one-hot encoded (pd.get_dummies); numeric columns used as-is
  (RandomForest does not require scaling).
- Primary evaluation: single stratified 75/25 train/test split (fixed random_state=42).
  RandomForestClassifier() with all-default hyperparameters trained twice on the
  training fold: once as-is, once on SMOTE-resampled (imblearn, default k_neighbors=5,
  random_state=42) training data. Both are evaluated on the *same* untouched test fold.
  Metric = F1 score with `>50K` as the positive class.
- Stability check: 20x repeated stratified 5-fold CV (4 repeats x 5 folds = 20 splits,
  varying random seed for both the CV split and SMOTE/RF internal randomness) to get a
  distribution of (F1_smote - F1_plain) differences, plus a bootstrap-style 95% CI on
  the mean difference across those folds.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

def load_and_encode(path="adult_income.csv"):
    df = pd.read_csv(path)
    df = df.copy()
    # pandas may represent text columns as dtype "object", "string", or (newer pandas) "str"
    cat_cols = [c for c in df.columns if df[c].dtype.name in ("object", "string", "str") and c != "class"]
    for c in cat_cols:
        df[c] = df[c].fillna("Missing")

    y = (df["class"].str.strip() == ">50K").astype(int)
    X = df.drop(columns=["class"])
    X = pd.get_dummies(X, columns=cat_cols, drop_first=False)
    return X, y

def main():
    X, y = load_and_encode()
    print("Data shape:", X.shape, "Positive rate:", y.mean())

    # ---------- Primary analysis: single 75/25 split ----------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )

    rf_plain = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_plain.fit(X_train, y_train)
    pred_plain = rf_plain.predict(X_test)
    f1_plain = f1_score(y_test, pred_plain, pos_label=1)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
    print("Train size before SMOTE:", X_train.shape, "class counts:", y_train.value_counts().to_dict())
    print("Train size after SMOTE:", X_train_sm.shape, "class counts:", pd.Series(y_train_sm).value_counts().to_dict())

    rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
    rf_smote.fit(X_train_sm, y_train_sm)
    pred_smote = rf_smote.predict(X_test)
    f1_smote = f1_score(y_test, pred_smote, pos_label=1)

    diff = f1_smote - f1_plain
    print(f"\nPrimary single-split result:")
    print(f"  F1 (no resampling) : {f1_plain:.4f}")
    print(f"  F1 (SMOTE)         : {f1_smote:.4f}")
    print(f"  Difference (SMOTE - plain): {diff:.4f}")

    # ---------- Stability check: repeated stratified 5-fold CV ----------
    n_splits = 5
    n_repeats = 4
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    diffs = []
    plain_scores = []
    smote_scores = []
    for i, (train_idx, test_idx) in enumerate(rskf.split(X, y)):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        seed = 1000 + i  # vary seed per fold for RF and SMOTE internal randomness

        rf_p = RandomForestClassifier(random_state=seed)
        rf_p.fit(X_tr, y_tr)
        f1_p = f1_score(y_te, rf_p.predict(X_te), pos_label=1)

        sm = SMOTE(random_state=seed)
        X_tr_sm, y_tr_sm = sm.fit_resample(X_tr, y_tr)
        rf_s = RandomForestClassifier(random_state=seed)
        rf_s.fit(X_tr_sm, y_tr_sm)
        f1_s = f1_score(y_te, rf_s.predict(X_te), pos_label=1)

        plain_scores.append(f1_p)
        smote_scores.append(f1_s)
        diffs.append(f1_s - f1_p)
        print(f"  fold {i+1:2d}/{n_splits*n_repeats}: F1_plain={f1_p:.4f}  F1_smote={f1_s:.4f}  diff={f1_s-f1_p:+.4f}")

    diffs = np.array(diffs)
    plain_scores = np.array(plain_scores)
    smote_scores = np.array(smote_scores)

    mean_diff = diffs.mean()
    std_diff = diffs.std(ddof=1)
    se_diff = std_diff / np.sqrt(len(diffs))
    ci_lo = mean_diff - 1.96 * se_diff
    ci_hi = mean_diff + 1.96 * se_diff

    print("\nRepeated CV summary:")
    print(f"  mean F1 plain : {plain_scores.mean():.4f} (sd={plain_scores.std(ddof=1):.4f})")
    print(f"  mean F1 smote : {smote_scores.mean():.4f} (sd={smote_scores.std(ddof=1):.4f})")
    print(f"  mean diff (smote - plain): {mean_diff:.4f}")
    print(f"  95% CI on mean diff: [{ci_lo:.4f}, {ci_hi:.4f}]")
    print(f"  |mean diff| > 0.02 ? {abs(mean_diff) > 0.02}")

    return {
        "f1_plain_single_split": f1_plain,
        "f1_smote_single_split": f1_smote,
        "diff_single_split": diff,
        "cv_mean_diff": mean_diff,
        "cv_ci_lo": ci_lo,
        "cv_ci_hi": ci_hi,
        "cv_plain_mean": plain_scores.mean(),
        "cv_smote_mean": smote_scores.mean(),
    }

if __name__ == "__main__":
    results = main()
    print("\nFinal results dict:", results)
