"""
H5: Does SMOTE oversampling change the minority-class (>50K) F1 score by more
than 0.02 compared to no resampling, holding the classifier fixed as a
default-hyperparameter RandomForestClassifier()?

Methodology
-----------
- Load adult_income.csv (48842 rows).
- Missing values (workclass, occupation, native-country contain NaN) are
  filled with the string 'Missing' so they become their own category
  rather than being dropped (preserves all 48842 rows).
- Categorical features one-hot encoded (pd.get_dummies), numeric features
  passed through as-is (RandomForest does not require scaling).
- Target 'class' mapped to binary: >50K = 1 (minority/positive class),
  <=50K = 0.
- Single stratified 80/20 train/test split (random_state=42 for
  reproducibility only, not a hyperparameter choice).
- Two RandomForestClassifier() models fit with ALL default hyperparameters
  except random_state=42 (for reproducibility):
    (a) trained directly on the raw (imbalanced) training data
    (b) trained on SMOTE-resampled training data (imblearn SMOTE,
        random_state=42, default k_neighbors=5)
  SMOTE is applied ONLY to the training split; the test split is left
  untouched (imbalanced, reflecting real-world class distribution) since
  resampling the evaluation set would give a biased/meaningless metric.
- Metric: F1 score of the positive/minority class ('>50K') on the held-out
  test set, i.e. f1_score(y_test, y_pred, pos_label=1).
- Effect: F1(SMOTE) - F1(no resampling). Compared to a threshold of 0.02
  in absolute value per the research question.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE

RANDOM_STATE = 42

df = pd.read_csv("adult_income.csv")

cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols = [c for c in cat_cols if c != "class"]
for c in cat_cols:
    df[c] = df[c].fillna("Missing")

y = (df["class"].str.strip() == ">50K").astype(int)
X = df.drop(columns=["class"])

X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=False)

X_train, X_test, y_train, y_test = train_test_split(
    X_enc, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)

print("Train class balance:\n", y_train.value_counts())
print("Test class balance:\n", y_test.value_counts())

# --- Model A: no resampling ---
rf_plain = RandomForestClassifier(random_state=RANDOM_STATE)
rf_plain.fit(X_train, y_train)
pred_plain = rf_plain.predict(X_test)
f1_plain = f1_score(y_test, pred_plain, pos_label=1)

print("\n=== No resampling ===")
print(classification_report(y_test, pred_plain, target_names=["<=50K", ">50K"]))
print("F1 (>50K):", f1_plain)

# --- Model B: SMOTE oversampling on training data only ---
smote = SMOTE(random_state=RANDOM_STATE)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
print("\nAfter SMOTE, train class balance:\n", y_train_sm.value_counts())

rf_smote = RandomForestClassifier(random_state=RANDOM_STATE)
rf_smote.fit(X_train_sm, y_train_sm)
pred_smote = rf_smote.predict(X_test)
f1_smote = f1_score(y_test, pred_smote, pos_label=1)

print("\n=== SMOTE resampling ===")
print(classification_report(y_test, pred_smote, target_names=["<=50K", ">50K"]))
print("F1 (>50K):", f1_smote)

diff = f1_smote - f1_plain
print("\nF1 difference (SMOTE - no resampling):", diff)
print("Exceeds 0.02 in absolute value:", abs(diff) > 0.02)
