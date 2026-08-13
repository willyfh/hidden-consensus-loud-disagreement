import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names and types:")
print(df.dtypes)
print("\nMissing values:")
print(df.isnull().sum())
print("\nTarget distribution:")
print(df['class'].value_counts())

# Data preprocessing
# Strip whitespace from all string columns
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.strip()

# Handle missing values - replace with 'Missing' for categorical, median for numeric
numeric_cols = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']

for col in categorical_cols:
    df[col] = df[col].fillna('Missing')

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col] = df[col].fillna(df[col].median())

# Prepare target variable
y = (df['class'] == '>50K').astype(int)

# Prepare features
X = df.drop(['class'], axis=1)

print("\n" + "="*60)
print("PREPROCESSING COMPLETE")
print("="*60)

# Encode categorical variables
X_encoded = X.copy()
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col])
    label_encoders[col] = le

print("\nEncoded features shape:", X_encoded.shape)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.3, random_state=42, stratify=y
)

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
print(f"Target balance in train: {y_train.sum()} / {len(y_train)} positive")

# Train Random Forest for feature importance
print("\n" + "="*60)
print("TRAINING RANDOM FOREST")
print("="*60)

rf = RandomForestClassifier(n_estimators=200, max_depth=20, n_jobs=-1, random_state=42)
rf.fit(X_train, y_train)

# Evaluate RF
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))
rf_train_auc = roc_auc_score(y_train, rf.predict_proba(X_train)[:, 1])
rf_test_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])

print(f"Random Forest - Train Accuracy: {rf_train_acc:.4f}, Test Accuracy: {rf_test_acc:.4f}")
print(f"Random Forest - Train AUC: {rf_train_auc:.4f}, Test AUC: {rf_test_auc:.4f}")

# Get feature importance from RF
feature_importance_rf = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("RANDOM FOREST FEATURE IMPORTANCE (Top 10)")
print("="*60)
print(feature_importance_rf.head(10).to_string())

# Train Logistic Regression for comparison
print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION")
print("="*60)

lr = LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42)
lr.fit(X_train, y_train)

# Evaluate LR
lr_train_acc = accuracy_score(y_train, lr.predict(X_train))
lr_test_acc = accuracy_score(y_test, lr.predict(X_test))
lr_train_auc = roc_auc_score(y_train, lr.predict_proba(X_train)[:, 1])
lr_test_auc = roc_auc_score(y_test, lr.predict_proba(X_test)[:, 1])

print(f"Logistic Regression - Train Accuracy: {lr_train_acc:.4f}, Test Accuracy: {lr_test_acc:.4f}")
print(f"Logistic Regression - Train AUC: {lr_train_auc:.4f}, Test AUC: {lr_test_auc:.4f}")

# Get feature importance from LR (absolute coefficients)
feature_importance_lr = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': np.abs(lr.coef_[0])
}).sort_values('importance', ascending=False)

print("\n" + "="*60)
print("LOGISTIC REGRESSION FEATURE IMPORTANCE (Top 10)")
print("="*60)
print(feature_importance_lr.head(10).to_string())

# Permutation importance on test set
print("\n" + "="*60)
print("COMPUTING PERMUTATION IMPORTANCE")
print("="*60)

from sklearn.inspection import permutation_importance

perm_importance = permutation_importance(
    rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
)

feature_importance_perm = pd.DataFrame({
    'feature': X_encoded.columns,
    'importance': perm_importance.importances_mean
}).sort_values('importance', ascending=False)

print("\nPermutation Importance (Top 10):")
print(feature_importance_perm.head(10).to_string())

# Summary
print("\n" + "="*60)
print("SUMMARY OF TOP FEATURES ACROSS METHODS")
print("="*60)

top_n = 5
print(f"\nTop {top_n} features by Random Forest importance:")
for i, row in feature_importance_rf.head(top_n).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print(f"\nTop {top_n} features by Permutation importance:")
for i, row in feature_importance_perm.head(top_n).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

print(f"\nTop {top_n} features by Logistic Regression coefficients:")
for i, row in feature_importance_lr.head(top_n).iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Identify the most important feature overall
# Use normalized scores
rf_normalized = feature_importance_rf.copy()
rf_normalized['importance'] = rf_normalized['importance'] / rf_normalized['importance'].sum()

perm_normalized = feature_importance_perm.copy()
perm_normalized['importance'] = perm_normalized['importance'] / perm_normalized['importance'].sum()

lr_normalized = feature_importance_lr.copy()
lr_normalized['importance'] = lr_normalized['importance'] / lr_normalized['importance'].sum()

# Merge and average
combined = rf_normalized.merge(perm_normalized, on='feature', suffixes=('_rf', '_perm'))
combined = combined.merge(lr_normalized[['feature', 'importance']], on='feature')
combined.rename(columns={'importance_rf': 'rf', 'importance_perm': 'perm', 'importance': 'lr'}, inplace=True)
combined['avg_importance'] = (combined['rf'] + combined['perm'] + combined['lr']) / 3
combined = combined.sort_values('avg_importance', ascending=False)

print("\n" + "="*60)
print("AVERAGED FEATURE IMPORTANCE (Top 10)")
print("="*60)
print(combined[['feature', 'rf', 'perm', 'lr', 'avg_importance']].head(10).to_string())

top_feature = combined.iloc[0]['feature']
top_importance = combined.iloc[0]['avg_importance']

print("\n" + "="*60)
print("CONCLUSION")
print("="*60)
print(f"\nMost important feature: {top_feature}")
print(f"Average normalized importance: {top_importance:.4f}")

# Additional insights
print("\nTop 3 most important features:")
for i, (idx, row) in enumerate(combined.head(3).iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['avg_importance']:.4f}")

# Check raw importance from RF model
print("\n\nTop 5 features by raw Random Forest importance:")
for i, (idx, row) in enumerate(feature_importance_rf.head(5).iterrows(), 1):
    print(f"{i}. {row['feature']}: {row['importance']:.4f} ({row['importance']*100:.2f}%)")
