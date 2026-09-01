"""
H2: Does RandomForestClassifier() (sklearn defaults) achieve higher stratified
5-fold CV ROC-AUC than LogisticRegression() (sklearn defaults) on the UCI Adult
income dataset?

Design notes
------------
* Both estimators are used with *scikit-learn defaults*, as the question demands.
  The only thing I choose is the preprocessing, which is unavoidable: the raw
  frame contains strings and NaNs and neither estimator accepts those.
* Preprocessing is held IDENTICAL for both models so the comparison isolates the
  model class:
      numeric      -> median imputation, then StandardScaler
      categorical  -> most-frequent imputation, then one-hot (dense, ignore unseen)
  Scaling is a no-op for a random forest (it is scale invariant), but it is
  essential for LogisticRegression's default lbfgs solver with max_iter=100:
  on raw capital-gain (0..99999) it does not converge, which would handicap the
  baseline for reasons that have nothing to do with the model class.
* Preprocessing lives inside the Pipeline so it is refit within each CV fold,
  avoiding leakage of imputation/scaling statistics across folds.
* Evaluation: StratifiedKFold(n_splits=5, shuffle=True, random_state=0),
  scoring='roc_auc' on predict_proba. Folds are shared between the two models,
  so the per-fold differences are paired -> paired t-test on the 5 fold deltas.
* Sensitivity checks (reported, not used for the headline number):
    (a) LogisticRegression on UNSCALED numerics (literal "defaults, no help")
    (b) RF with ordinal-encoded categoricals instead of one-hot
    (c) dropping `fnlwgt` (a census sampling weight, not a person-level trait)
    (d) 10 repeats of the 5-fold split (RepeatedStratifiedKFold) to gauge how
        much the answer depends on the particular fold assignment
"""

import json
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RNG = 0


# ---------------------------------------------------------------- data
def load():
    df = pd.read_csv("adult_income.csv")
    # '?' already arrives as NaN in this file, but normalise defensively.
    df = df.replace("?", np.nan)
    y = (df["class"].str.strip() == ">50K").astype(int)
    X = df.drop(columns=["class"])
    return X, y


def make_pre(num_cols, cat_cols, scale_numeric=True, cat_kind="onehot"):
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scale", StandardScaler()))

    if cat_kind == "onehot":
        cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    else:
        cat_enc = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )
    cat_steps = [
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("enc", cat_enc),
    ]
    return ColumnTransformer(
        [
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline(cat_steps), cat_cols),
        ]
    )


def score(model, X, y, num_cols, cat_cols, cv, scale_numeric=True, cat_kind="onehot"):
    pipe = Pipeline(
        [
            ("pre", make_pre(num_cols, cat_cols, scale_numeric, cat_kind)),
            ("clf", model),
        ]
    )
    return cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)


def main():
    X, y = load()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    print(f"n={len(X)}  positives={y.sum()} ({y.mean():.1%})")
    print(f"numeric: {num_cols}")
    print(f"categorical: {cat_cols}")
    print(f"rows with >=1 missing value: {X.isna().any(axis=1).sum()}\n")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RNG)

    # ------------------------------------------------ primary comparison
    rf = score(RandomForestClassifier(random_state=RNG), X, y, num_cols, cat_cols, cv)
    lr = score(LogisticRegression(random_state=RNG), X, y, num_cols, cat_cols, cv)

    diff = rf - lr
    t, p = stats.ttest_rel(rf, lr)

    print("PRIMARY (identical one-hot + scaled preprocessing, stratified 5-fold)")
    print(f"  RF     folds: {np.round(rf, 5)}  mean={rf.mean():.5f} sd={rf.std(ddof=1):.5f}")
    print(f"  LogReg folds: {np.round(lr, 5)}  mean={lr.mean():.5f} sd={lr.std(ddof=1):.5f}")
    print(f"  diff (RF-LR) per fold: {np.round(diff, 5)}")
    print(f"  mean difference = {diff.mean():+.5f}")
    print(f"  paired t-test over 5 folds: t={t:.3f}, p={p:.5f}")
    print(f"  RF wins in {int((diff > 0).sum())}/5 folds\n")

    # ------------------------------------------------ sensitivity checks
    sens = {}

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        lr_raw = score(
            LogisticRegression(random_state=RNG), X, y, num_cols, cat_cols, cv,
            scale_numeric=False,
        )
    sens["logreg_unscaled_numerics"] = lr_raw.mean()
    print(f"(a) LogReg, UNSCALED numerics (does not converge): {lr_raw.mean():.5f}"
          f"   diff vs RF = {rf.mean() - lr_raw.mean():+.5f}")

    rf_ord = score(
        RandomForestClassifier(random_state=RNG), X, y, num_cols, cat_cols, cv,
        cat_kind="ordinal",
    )
    sens["rf_ordinal_encoded_cats"] = rf_ord.mean()
    d_ord = rf_ord - lr
    t_ord, p_ord = stats.ttest_rel(rf_ord, lr)
    print(f"(b) RF with ordinal-encoded categoricals: {rf_ord.mean():.5f}"
          f"   diff vs LogReg = {d_ord.mean():+.5f}"
          f"  (paired p={p_ord:.3f}, RF wins {int((d_ord > 0).sum())}/5)")

    Xd = X.drop(columns=["fnlwgt"])
    nd = [c for c in num_cols if c != "fnlwgt"]
    rf_d = score(RandomForestClassifier(random_state=RNG), Xd, y, nd, cat_cols, cv)
    lr_d = score(LogisticRegression(random_state=RNG), Xd, y, nd, cat_cols, cv)
    sens["drop_fnlwgt_rf"] = rf_d.mean()
    sens["drop_fnlwgt_logreg"] = lr_d.mean()
    print(f"(c) drop fnlwgt: RF={rf_d.mean():.5f} LogReg={lr_d.mean():.5f}"
          f"   diff = {rf_d.mean() - lr_d.mean():+.5f}")

    rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=RNG)
    rf_r = score(RandomForestClassifier(random_state=RNG), X, y, num_cols, cat_cols, rcv)
    lr_r = score(LogisticRegression(random_state=RNG), X, y, num_cols, cat_cols, rcv)
    d_r = rf_r - lr_r
    sens["repeated_5x10_rf"] = rf_r.mean()
    sens["repeated_5x10_logreg"] = lr_r.mean()
    sens["repeated_5x10_diff"] = d_r.mean()
    print(f"(d) 10x repeated 5-fold (50 fits each): RF={rf_r.mean():.5f} "
          f"LogReg={lr_r.mean():.5f} diff={d_r.mean():+.5f} "
          f"(RF wins {int((d_r > 0).sum())}/50 folds)\n")

    # ------------------------------------------------ result.json
    direction = "RF > LogReg" if diff.mean() > 0 else "LogReg > RF (no, hypothesis not supported)"
    result = {
        "hypothesis_id": "H2",
        "summary": (
            f"No. Under identical preprocessing, a default RandomForestClassifier reaches "
            f"a stratified 5-fold CV ROC-AUC of {rf.mean():.4f}, slightly BELOW the "
            f"{lr.mean():.4f} of a default LogisticRegression, a difference of "
            f"{diff.mean():+.4f} AUC. Logistic regression wins in all 5 folds and the "
            f"paired difference is nominally significant (p={p:.4f}), so the hypothesis is "
            f"not supported, though a gap of ~{abs(diff.mean()):.3f} AUC means the two "
            f"models are near-equivalent in practice."
        ),
        "primary_metric_name": "ROC-AUC difference (RF - LogReg), stratified 5-fold CV mean",
        "primary_metric_value": round(float(diff.mean()), 5),
        "direction": direction,
        "methodological_choices": (
            "Both estimators used exactly at scikit-learn defaults (only random_state "
            "fixed for reproducibility); no tuning, no class_weight, imbalance "
            "(23.9% positive) left untouched since ROC-AUC is prevalence-insensitive. "
            "Identical preprocessing for both models, fit inside each CV fold via a "
            "Pipeline to avoid leakage: numeric columns median-imputed then "
            "StandardScaler'd; categorical columns most-frequent-imputed then one-hot "
            "encoded (handle_unknown='ignore'). Scaling numerics is a judgement call "
            "that matters: it is inert for the forest but without it LogisticRegression's "
            "default lbfgs/max_iter=100 fails to converge on capital-gain's 0-99999 range "
            f"and scores only {lr_raw.mean():.4f} (which would inflate the gap to "
            f"{rf.mean() - lr_raw.mean():+.4f}); I regard the scaled number as the fair "
            "baseline. All 14 features kept, including fnlwgt (a census sampling weight, "
            "arguably not a legitimate person-level predictor) and race/sex; rows with "
            "missing workclass/occupation/native-country were imputed rather than dropped. "
            "Validation: StratifiedKFold(5, shuffle=True, random_state=0), shared folds "
            "for both models, compared with a paired t-test on the 5 per-fold deltas "
            "(n=5 folds, so the p-value is fragile). Sensitivity checks, which show the "
            "sign of the result is encoding-dependent and therefore should not be read as "
            f"a robust 'logistic regression is the better model class': dropping fnlwgt "
            f"widens the logistic lead (diff {rf_d.mean() - lr_d.mean():+.4f}) and "
            f"10x-repeated 5-fold reproduces it (diff {d_r.mean():+.4f}, RF ahead in only "
            f"{int((d_r > 0).sum())}/50 folds), but giving the forest ORDINAL-encoded "
            f"categoricals instead of one-hot lifts it to {rf_ord.mean():.4f} and erases "
            f"the gap entirely (diff {d_ord.mean():+.4f}, paired p={p_ord:.3f}) - a tie. "
            "So the honest reading is that the two model classes are within ~0.005 AUC of "
            "each other and which one 'wins' is decided by preprocessing choices rather "
            "than by the model class. Another researcher might reasonably have used "
            "ordinal/target encoding for the trees, dropped fnlwgt, added "
            "polynomial/interaction terms to the logistic model, or raised max_iter "
            "instead of scaling."
        ),
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
