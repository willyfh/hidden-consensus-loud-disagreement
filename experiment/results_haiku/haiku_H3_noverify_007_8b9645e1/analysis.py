import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

# Load data
df = pd.read_csv('adult_income.csv')
print(f"Dataset shape: {df.shape}")
print(f"\nColumn names and types:")
print(df.dtypes)
print(f"\nFirst few rows:")
print(df.head())
print(f"\nTarget distribution:")
print(df['class'].value_counts())

# Data exploration
print("\n" + "="*60)
print("DATA EXPLORATION")
print("="*60)

# Check missing values
print(f"\nMissing values:")
missing = df.isnull().sum()
print(missing[missing > 0])

# Identify feature types
categorical_features = df.select_dtypes(include='object').columns.tolist()
categorical_features.remove('class')  # Remove target
numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")
print(f"Numeric features ({len(numeric_features)}): {numeric_features}")

# Remove rows with missing values in key features
df_clean = df.dropna(subset=['workclass', 'occupation'])

# Separate features and target
X = df_clean.drop('class', axis=1)
y = df_clean['class']

# Encode target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\nAfter dropping missing values, shape: {X.shape}")
print(f"Target encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
        ('num', 'passthrough', numeric_features)
    ]
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names after encoding
cat_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate([cat_feature_names, numeric_features])

print(f"\nProcessed feature count: {X_train_processed.shape[1]}")

# Train models
print("\n" + "="*60)
print("MODEL TRAINING AND FEATURE IMPORTANCE")
print("="*60)

# Gradient Boosting Classifier (best for feature importance)
print("\n1. Gradient Boosting Classifier:")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42, max_depth=5)
gb_model.fit(X_train_processed, y_train)
gb_score = gb_model.score(X_test_processed, y_test)
print(f"   Test accuracy: {gb_score:.4f}")

gb_importance = gb_model.feature_importances_
gb_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': gb_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 15 features by Gradient Boosting importance:")
print(gb_importance_df.head(15).to_string(index=False))

# Random Forest for comparison
print("\n2. Random Forest Classifier:")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15)
rf_model.fit(X_train_processed, y_train)
rf_score = rf_model.score(X_test_processed, y_test)
print(f"   Test accuracy: {rf_score:.4f}")

rf_importance = rf_model.feature_importances_
rf_importance_df = pd.DataFrame({
    'feature': all_feature_names,
    'importance': rf_importance
}).sort_values('importance', ascending=False)

print(f"\nTop 15 features by Random Forest importance:")
print(rf_importance_df.head(15).to_string(index=False))

# Group importance by original feature (before one-hot encoding)
print("\n" + "="*60)
print("FEATURE IMPORTANCE BY ORIGINAL FEATURE")
print("="*60)

feature_groups = {}

# Group categorical one-hot encoded features
for cat_feature in categorical_features:
    related_cols = [col for col in gb_importance_df['feature'] if col.startswith(cat_feature)]
    if related_cols:
        group_importance = gb_importance_df[gb_importance_df['feature'].isin(related_cols)]['importance'].sum()
        feature_groups[cat_feature] = group_importance

# Add numeric features
for num_feature in numeric_features:
    if num_feature in gb_importance_df['feature'].values:
        importance = gb_importance_df[gb_importance_df['feature'] == num_feature]['importance'].values
        if len(importance) > 0:
            feature_groups[num_feature] = importance[0]

grouped_importance = pd.DataFrame({
    'feature': list(feature_groups.keys()),
    'importance': list(feature_groups.values())
}).sort_values('importance', ascending=False)

print(f"\nFeature importance by original feature (Gradient Boosting):")
print(grouped_importance.to_string(index=False))

# Analysis of top features
print("\n" + "="*60)
print("TOP FEATURES ANALYSIS")
print("="*60)

top_3 = grouped_importance.head(3)
print(f"\nTop 3 most important features:")
for idx, row in top_3.iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# Check how original features map to importance
print("\n" + "="*60)
print("DETAILED BREAKDOWN OF TOP FEATURES")
print("="*60)

for feature in top_3['feature'].values[:5]:
    print(f"\n{feature}:")
    if feature in numeric_features:
        print(f"  Type: Numeric")
    else:
        print(f"  Type: Categorical")

    # Show contribution from one-hot components if categorical
    related = gb_importance_df[gb_importance_df['feature'].str.startswith(feature)]
    if len(related) > 0:
        print(f"  Individual category importances:")
        for idx, row in related.head(5).iterrows():
            print(f"    {row['feature']}: {row['importance']:.4f}")

# Summary statistics
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"\nTotal features in dataset: {len(X.columns)}")
print(f"Features analyzed: {len(all_feature_names)}")
print(f"Model accuracy (GB): {gb_score:.4f}")
print(f"Model accuracy (RF): {rf_score:.4f}")

top_feature = grouped_importance.iloc[0]
print(f"\nMost important feature: {top_feature['feature']}")
print(f"Importance score: {top_feature['importance']:.4f}")
print(f"Percentage of total importance: {(top_feature['importance']/grouped_importance['importance'].sum())*100:.2f}%")

# Save results to result.json
import json

result = {
    "hypothesis_id": "H3",
    "summary": f"Using Gradient Boosting classification with one-hot encoded categorical features and numeric features, the most important feature for predicting income is {top_feature['feature']} with an importance score of {top_feature['importance']:.4f}. The top 3 features are {top_feature['feature']}, {grouped_importance.iloc[1]['feature']}, and {grouped_importance.iloc[2]['feature']}, which together account for {(grouped_importance.head(3)['importance'].sum()/grouped_importance['importance'].sum())*100:.1f}% of the model's predictive power.",
    "primary_metric_name": "Top feature importance (Gradient Boosting)",
    "primary_metric_value": float(top_feature['importance']),
    "direction": f"{top_feature['feature']} is most important",
    "methodological_choices": "Gradient Boosting Classifier (100 estimators, max_depth=5) trained on preprocessed data with one-hot encoding for 13 categorical features and 1 numeric feature (fnlwgt). Rows with missing values in workclass and occupation were removed. Target was binary encoded (<=50K=0, >50K=1). 80-20 train-test split with stratification. Feature importance calculated from tree-based model's native importance scores. Original features grouped by name to aggregate one-hot encoded components."
}

with open('result.json', 'w') as f:
    json.dump(result, f, indent=2)

print("\nResults saved to result.json")
print(json.dumps(result, indent=2))
