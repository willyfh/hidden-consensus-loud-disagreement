"""
H5: Does SMOTE oversampling of the training data change the minority-class (>50K)
F1 score by more than 0.02, compared to no resampling, holding the classifier fixed
as a default-hyperparameter RandomForestClassifier()?
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

def load_data(path="adult_income.csv"):
    df = pd.read_csv(path)
    # Missing values appear only in workclass, occupation, native-country (all categorical,
    # encoded as NaN from '?' originally). Treat as its own category "Missing" rather than
    # dropping rows, to preserve sample size / avoid biasing the minority class.
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    cat_cols = [c for c in cat_cols if c != "class"]
    for c in cat_cols:
        df[c] = df[c].fillna("Missing")

    y = (df["class"] == ">50K").astype(int)
    X = df.drop(columns=["class"])

    # Ordinal-encode categoricals. RandomForest doesn't need one-hot; ordinal encoding
    # keeps dimensionality low and is a defensible, common choice for tree models.
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X[cat_cols] = enc.fit_transform(X[cat_cols])
    return X, y


def run_single_split(X, y, random_state=RANDOM_STATE, test_size=0.25):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # No resampling
    clf_plain = RandomForestClassifier(random_state=random_state)
    clf_plain.fit(X_train, y_train)
    pred_plain = clf_plain.predict(X_test)
    f1_plain = f1_score(y_test, pred_plain, pos_label=1)

    # SMOTE on training data only
    sm = SMOTE(random_state=random_state)
    X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
    clf_smote = RandomForestClassifier(random_state=random_state)
    clf_smote.fit(X_train_sm, y_train_sm)
    pred_smote = clf_smote.predict(X_test)
    f1_smote = f1_score(y_test, pred_smote, pos_label=1)

    return f1_plain, f1_smote


def run_repeated_cv(X, y, n_splits=5, n_repeats=5, base_seed=100):
    """Repeated stratified k-fold with different seeds each repeat, for stability check."""
    plain_scores = []
    smote_scores = []
    for r in range(n_repeats):
        seed = base_seed + r
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            clf_plain = RandomForestClassifier(random_state=seed)
            clf_plain.fit(X_train, y_train)
            f1_plain = f1_score(y_test, clf_plain.predict(X_test), pos_label=1)
            plain_scores.append(f1_plain)

            sm = SMOTE(random_state=seed)
            X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)
            clf_smote = RandomForestClassifier(random_state=seed)
            clf_smote.fit(X_train_sm, y_train_sm)
            f1_smote = f1_score(y_test, clf_smote.predict(X_test), pos_label=1)
            smote_scores.append(f1_smote)

    return np.array(plain_scores), np.array(smote_scores)


def main():
    X, y = load_data()
    print("Data shape:", X.shape)
    print("Class balance:\n", y.value_counts(normalize=True))

    print("\n--- Primary single train/test split (75/25, stratified, seed=42) ---")
    f1_plain, f1_smote = run_single_split(X, y)
    diff = f1_smote - f1_plain
    print(f"F1 (no resampling): {f1_plain:.4f}")
    print(f"F1 (SMOTE):         {f1_smote:.4f}")
    print(f"Difference (SMOTE - plain): {diff:.4f}")

    print("\n--- Verification: 5x repeated 5-fold stratified CV, different seeds ---")
    plain_scores, smote_scores = run_repeated_cv(X, y)
    diffs = smote_scores - plain_scores
    print(f"No-resampling F1: mean={plain_scores.mean():.4f} std={plain_scores.std():.4f}")
    print(f"SMOTE F1:         mean={smote_scores.mean():.4f} std={smote_scores.std():.4f}")
    print(f"Mean diff (SMOTE - plain): {diffs.mean():.4f}")
    print(f"Std of diff: {diffs.std():.4f}")
    print(f"Min/Max diff across folds: {diffs.min():.4f} / {diffs.max():.4f}")
    # 95% CI via normal approx on the paired differences
    n = len(diffs)
    se = diffs.std(ddof=1) / np.sqrt(n)
    ci_low = diffs.mean() - 1.96 * se
    ci_high = diffs.mean() + 1.96 * se
    print(f"95% CI of mean diff: [{ci_low:.4f}, {ci_high:.4f}]")

    result = {
        "primary_split_f1_plain": f1_plain,
        "primary_split_f1_smote": f1_smote,
        "primary_split_diff": diff,
        "cv_diff_mean": float(diffs.mean()),
        "cv_diff_std": float(diffs.std()),
        "cv_diff_ci_low": float(ci_low),
        "cv_diff_ci_high": float(ci_high),
        "cv_plain_mean": float(plain_scores.mean()),
        "cv_smote_mean": float(smote_scores.mean()),
    }
    return result


if __name__ == "__main__":
    main()
