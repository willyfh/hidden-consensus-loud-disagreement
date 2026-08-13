"""
H2: Does RandomForestClassifier() outperform LogisticRegression() (sklearn defaults)
on stratified 5-fold CV ROC-AUC for the Adult Income dataset?
"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import roc_auc_score

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load & prepare data
# ---------------------------------------------------------------------------
df = pd.read_csv("adult_income.csv")
df = df.drop_duplicates().reset_index(drop=True)

target = "class"
y = (df[target] == ">50K").astype(int)
X = df.drop(columns=[target])

# fnlwgt is a census sampling weight, not a real demographic feature of the
# individual -- drop it, it's not meaningful as a predictor of income class.
X = X.drop(columns=["fnlwgt"])

num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
print("Numeric cols:", num_cols)
print("Categorical cols:", cat_cols)

# Missing values in categoricals (workclass, occupation, native-country) are
# imputed with a constant "Missing" category rather than dropped, to avoid
# losing ~7% of rows and because "missingness" itself may carry signal.
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, cat_cols),
])

logreg_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", LogisticRegression()),  # sklearn defaults as specified
])
rf_pipe = Pipeline(steps=[
    ("preprocess", preprocess),
    ("clf", RandomForestClassifier(random_state=RANDOM_STATE)),  # defaults + fixed seed for reproducibility
])

# ---------------------------------------------------------------------------
# 2. Primary analysis: stratified 5-fold CV ROC-AUC
# ---------------------------------------------------------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

logreg_scores = cross_val_score(logreg_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
rf_scores = cross_val_score(rf_pipe, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)

print("\n=== Primary: single stratified 5-fold CV (seed=42) ===")
print("LogReg AUC per fold:", np.round(logreg_scores, 4), "mean:", logreg_scores.mean().round(4))
print("RF     AUC per fold:", np.round(rf_scores, 4), "mean:", rf_scores.mean().round(4))
primary_diff = rf_scores.mean() - logreg_scores.mean()
print("Mean AUC diff (RF - LogReg):", round(primary_diff, 4))

# ---------------------------------------------------------------------------
# 3. Stability check A: 5x repeated stratified 5-fold CV, multiple seeds
# ---------------------------------------------------------------------------
rcv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=123)
logreg_rep = cross_val_score(logreg_pipe, X, y, cv=rcv, scoring="roc_auc", n_jobs=-1)
rf_rep = cross_val_score(rf_pipe, X, y, cv=rcv, scoring="roc_auc", n_jobs=-1)

print("\n=== Verification: 5x repeated stratified 5-fold CV (25 folds total, seed=123) ===")
print(f"LogReg: mean={logreg_rep.mean():.4f} std={logreg_rep.std():.4f}")
print(f"RF:     mean={rf_rep.mean():.4f} std={rf_rep.std():.4f}")
rep_diff = rf_rep.mean() - logreg_rep.mean()
print("Mean AUC diff (RF - LogReg):", round(rep_diff, 4))

# paired diff per fold (since same folds used for both models within each repeat)
paired_diffs = rf_rep - logreg_rep
ci_lower = np.percentile(paired_diffs, 2.5)
ci_upper = np.percentile(paired_diffs, 97.5)
print(f"Paired fold-level diff: mean={paired_diffs.mean():.4f}, "
      f"95% range [{ci_lower:.4f}, {ci_upper:.4f}], "
      f"RF wins {np.mean(paired_diffs > 0)*100:.0f}% of folds")

# ---------------------------------------------------------------------------
# 4. Stability check B: independent held-out re-test split (not used above)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=999
)
logreg_pipe.fit(X_train, y_train)
rf_pipe.fit(X_train, y_train)
logreg_test_auc = roc_auc_score(y_test, logreg_pipe.predict_proba(X_test)[:, 1])
rf_test_auc = roc_auc_score(y_test, rf_pipe.predict_proba(X_test)[:, 1])

print("\n=== Verification: independent 70/30 held-out split (seed=999) ===")
print(f"LogReg test AUC: {logreg_test_auc:.4f}")
print(f"RF     test AUC: {rf_test_auc:.4f}")
print(f"Diff (RF - LogReg): {rf_test_auc - logreg_test_auc:.4f}")

# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------
print("\n=== SUMMARY ===")
print(f"Primary 5-fold CV:      LogReg={logreg_scores.mean():.4f}  RF={rf_scores.mean():.4f}  diff={primary_diff:.4f}")
print(f"Repeated CV (25 folds): LogReg={logreg_rep.mean():.4f}  RF={rf_rep.mean():.4f}  diff={rep_diff:.4f}")
print(f"Held-out re-test:       LogReg={logreg_test_auc:.4f}  RF={rf_test_auc:.4f}  diff={rf_test_auc - logreg_test_auc:.4f}")
