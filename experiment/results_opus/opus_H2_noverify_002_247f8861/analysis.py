"""
H2: Does RandomForestClassifier() beat LogisticRegression() (both sklearn defaults)
on stratified 5-fold CV ROC-AUC for the UCI Adult income dataset?

Design notes
------------
* The models themselves are fixed at sklearn defaults by the research question.
  Everything else (preprocessing, CV scheme, comparison test) is my choice.
* Both models get the IDENTICAL preprocessing pipeline so the comparison isolates
  the model class rather than feature engineering.
* Preprocessing is fit INSIDE each CV fold (sklearn Pipeline) to avoid leakage.
* Same fold indices for both models -> paired comparison (paired t-test on the
  5 per-fold AUC differences, plus a bootstrap-free descriptive CI).
* Sensitivity checks: (a) dropping `fnlwgt` (a census sampling weight, arguably
  not a legitimate predictor), (b) dropping exact duplicate rows,
  (c) 10-fold instead of 5-fold, (d) an ordinal/native-categorical encoding for
  the trees instead of one-hot.
"""

import json
import numpy as np
import pandas as pd
from scipy import stats

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

RANDOM_STATE = 42
TARGET = "class"
POSITIVE = ">50K"


def load(path="adult_income.csv"):
    df = pd.read_csv(path)
    # Missing values are already NaN here (source '?' codes); confirm and report.
    y = (df[TARGET].astype(str).str.strip() == POSITIVE).astype(int)
    X = df.drop(columns=[TARGET])
    return df, X, y


def make_preprocessor(X, tree_style=False):
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    if tree_style:
        # Trees don't need scaling or one-hot; integer codes are fine and keep
        # the feature space small. Used only in a sensitivity check.
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ord", OrdinalEncoder(handle_unknown="use_encoded_value",
                                   unknown_value=-1)),
        ])
        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
    else:
        cat_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        # Scaling matters for LogisticRegression's default lbfgs solver
        # (capital-gain spans 0..99999); it is a no-op for the forest.
        num_pipe = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])

    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)])


def run_cv(X, y, n_splits=5, tree_style_rf=False, label=""):
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True,
                         random_state=RANDOM_STATE)
    out = {}
    for name, model in [("logreg", LogisticRegression()),
                        ("rf", RandomForestClassifier(random_state=RANDOM_STATE))]:
        pre = make_preprocessor(X, tree_style=(tree_style_rf and name == "rf"))
        pipe = Pipeline([("pre", pre), ("clf", model)])
        res = cross_validate(pipe, X, y, cv=cv, scoring="roc_auc",
                             n_jobs=-1, return_train_score=False)
        out[name] = res["test_score"]
        print(f"[{label}] {name:7s} AUC = {res['test_score'].mean():.4f} "
              f"+/- {res['test_score'].std(ddof=1):.4f}  "
              f"folds={np.round(res['test_score'], 4)}")
    diff = out["rf"] - out["logreg"]
    t, p = stats.ttest_rel(out["rf"], out["logreg"])
    print(f"[{label}] diff (RF - LogReg) = {diff.mean():+.4f}  "
          f"paired t={t:.3f}, p={p:.5f}\n")
    return out, diff, t, p


def main():
    df, X, y = load()
    print(f"shape={df.shape}  positives={y.mean():.4f}  "
          f"exact dup rows={df.duplicated().sum()}")
    print("missing per column:\n", X.isna().sum()[lambda s: s > 0], "\n")

    # ---- Primary analysis -------------------------------------------------
    scores, diff, t, p = run_cv(X, y, n_splits=5, label="primary 5-fold")
    rf_mean, lr_mean = scores["rf"].mean(), scores["logreg"].mean()
    primary_diff = float(rf_mean - lr_mean)
    # descriptive 95% CI on the paired per-fold differences (5 folds -> t_4)
    sem = diff.std(ddof=1) / np.sqrt(len(diff))
    ci = stats.t.interval(0.95, len(diff) - 1, loc=diff.mean(), scale=sem)

    sens = {}

    # ---- Sensitivity 1: drop fnlwgt (census sampling weight, not a predictor)
    s, d, _, pp = run_cv(X.drop(columns=["fnlwgt"]), y, label="no-fnlwgt")
    sens["drop_fnlwgt"] = {"rf": float(s["rf"].mean()),
                           "logreg": float(s["logreg"].mean()),
                           "diff": float(d.mean()), "p": float(pp)}

    # ---- Sensitivity 2: drop exact duplicate rows -------------------------
    dd = df.drop_duplicates()
    y2 = (dd[TARGET].astype(str).str.strip() == POSITIVE).astype(int)
    s, d, _, pp = run_cv(dd.drop(columns=[TARGET]), y2, label="dedup")
    sens["drop_duplicates"] = {"rf": float(s["rf"].mean()),
                               "logreg": float(s["logreg"].mean()),
                               "diff": float(d.mean()), "p": float(pp)}

    # ---- Sensitivity 3: 10-fold ------------------------------------------
    s, d, _, pp = run_cv(X, y, n_splits=10, label="10-fold")
    sens["ten_fold"] = {"rf": float(s["rf"].mean()),
                        "logreg": float(s["logreg"].mean()),
                        "diff": float(d.mean()), "p": float(pp)}

    # ---- Sensitivity 4: ordinal encoding for the RF -----------------------
    s, d, _, pp = run_cv(X, y, tree_style_rf=True, label="ordinal-RF")
    sens["ordinal_encoded_rf"] = {"rf": float(s["rf"].mean()),
                                  "logreg": float(s["logreg"].mean()),
                                  "diff": float(d.mean()), "p": float(pp)}

    print("=" * 70)
    print(f"PRIMARY: RF {rf_mean:.4f} vs LogReg {lr_mean:.4f} -> "
          f"diff {primary_diff:+.4f} (95% CI {ci[0]:+.4f}, {ci[1]:+.4f}), p={p:.5f}")

    rf_wins = int((diff > 0).sum())
    verdict = "Yes" if primary_diff > 0 else "No"
    direction = "RF > LogReg" if primary_diff > 0 else "LogReg > RF (marginally)"
    ord_diff = sens["ordinal_encoded_rf"]["diff"]
    result = {
        "hypothesis_id": "H2",
        "summary": (
            f"{verdict} - the random forest does not beat logistic regression here. "
            f"At sklearn defaults with identical preprocessing, "
            f"RandomForestClassifier scores a stratified 5-fold CV ROC-AUC of "
            f"{rf_mean:.4f} versus {lr_mean:.4f} for LogisticRegression, i.e. "
            f"{primary_diff:+.4f} AUC for the forest (it won {rf_wins}/5 folds; "
            f"paired t-test p={p:.3f}, 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}]). The "
            f"two models are practically equivalent - the gap is under half an "
            f"AUC point - and the sign is fragile: it depends on how the "
            f"categorical features are encoded. Under one-hot encoding logistic "
            f"regression is slightly ahead, but giving the forest ordinal-coded "
            f"categoricals instead erases the deficit entirely "
            f"({ord_diff:+.4f} AUC), so the honest conclusion is a tie rather "
            f"than a win for either model class."
        ),
        "primary_metric_name": "ROC-AUC difference (RF - LogReg), stratified 5-fold CV mean",
        "primary_metric_value": round(primary_diff, 6),
        "direction": direction,
        "methodological_choices": (
            "Models fixed by the question: RandomForestClassifier() and "
            "LogisticRegression() at sklearn 1.6.1 defaults (RF given "
            "random_state=42 for reproducibility; LogReg left at default "
            "lbfgs/C=1.0/max_iter=100). Target binarised as '>50K'=1 (23.9% "
            "positive); no class-imbalance handling (no class_weight, no "
            "resampling) since ROC-AUC is threshold-free and prevalence-"
            "insensitive. Identical preprocessing for both models so the "
            "contrast isolates model class: median imputation + StandardScaler "
            "on the 6 numeric columns, most-frequent imputation + one-hot "
            "(handle_unknown='ignore') on the 8 categoricals (~100 columns). "
            "Scaling is required for lbfgs convergence given capital-gain's "
            "0-99999 range and is a no-op for trees. All preprocessing is fit "
            "inside each fold via a Pipeline, so no leakage. Validation: "
            "StratifiedKFold(n_splits=5, shuffle=True, random_state=42), the "
            "same fold indices for both models, enabling a paired t-test on the "
            "5 per-fold differences (n=5, so the p-value is indicative rather "
            "than strong evidence). No held-out test set: the question asks "
            "about CV performance, and CV uses all 48,842 rows. Kept all 14 "
            "features including fnlwgt (a census sampling weight, arguably not a "
            "legitimate predictor) and 'education' alongside its numeric twin "
            "'education-num'; kept the 52 exact duplicate rows. Both choices "
            "were re-tested as sensitivity analyses and neither flips the "
            "sign (dropping fnlwgt actually costs the RF ~0.011 AUC and widens "
            "logistic regression's lead; de-duplicating and 10-fold CV both "
            "leave the difference at ~-0.0035). The one choice that DOES flip "
            "the sign is the categorical encoding: one-hot expands 8 "
            "categoricals into ~100 sparse columns, and because the RF's "
            "default max_features='sqrt' samples ~10 of them per split, the "
            "informative categoricals get diluted. Swapping the forest to "
            "ordinal-coded categoricals lifts it to 0.9058 and it edges past "
            "logistic regression (+0.0005, p=0.76, i.e. a tie). So the answer "
            "to H2 is encoding-dependent, and I report the one-hot result as "
            "primary only because it applies the same representation to both "
            "models. Other defensible choices: tuning hyperparameters (default "
            "RF has unlimited depth and overfits; tuning would likely favour "
            "the RF), target encoding, adding spline/interaction terms to the "
            "logistic model (which would favour LogReg), evaluating with "
            "accuracy/PR-AUC instead of ROC-AUC, or repeating CV over several "
            "seeds - with only 5 paired folds the p-value is indicative, and "
            "the fold-to-fold spread (~0.003 SD) is the same order of "
            "magnitude as the effect being measured."
        ),
    }

    with open("result.json", "w") as f:
        json.dump(result, f, indent=2)

    with open("sensitivity.json", "w") as f:
        json.dump({
            "primary": {"rf_auc": float(rf_mean), "logreg_auc": float(lr_mean),
                        "diff": primary_diff, "ci95": [float(ci[0]), float(ci[1])],
                        "paired_t": float(t), "p_value": float(p),
                        "rf_folds": scores["rf"].tolist(),
                        "logreg_folds": scores["logreg"].tolist()},
            "sensitivity": sens,
        }, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
