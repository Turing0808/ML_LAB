import warnings
warnings.filterwarnings("ignore")  # Suppress all warnings

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
import lime.lime_tabular

def load_and_split(filepath):
    """Load dataset and split into train/test, cleaning mixed-type issues"""
    data = pd.read_csv(filepath, low_memory=False)
    data = data.dropna(subset=['class1'])

    # Drop problematic columns (mixed types)
    if 'duration' in data.columns:
        data = data.drop(columns=['duration'])

    X = data.drop(columns=['class1'])
    y = LabelEncoder().fit_transform(data['class1'])

    return train_test_split(X, y, test_size=0.2, random_state=42), X.columns

def build_pipeline():
    """Build a simple Random Forest pipeline"""
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

def main():
    filepath = r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\cleaned.csv"
    (X_train, X_test, y_train, y_test), feature_names = load_and_split(filepath)

    # Convert numeric columns properly
    numeric_cols = X_train.select_dtypes(include=['float64', 'int64']).columns
    X_train[numeric_cols] = X_train[numeric_cols].apply(pd.to_numeric, errors='coerce')
    X_test[numeric_cols] = X_test[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Fill missing values for LIME
    X_train[numeric_cols] = X_train[numeric_cols].fillna(X_train[numeric_cols].mean())
    X_test[numeric_cols] = X_test[numeric_cols].fillna(X_train[numeric_cols].mean())

    # Convert categorical columns to strings
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)

    categorical_features = [X_train.columns.get_loc(col) for col in categorical_cols]

    # Build and train pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=np.array(X_train),
        feature_names=feature_names,
        class_names=[str(c) for c in np.unique(y_train)],
        categorical_features=categorical_features,
        mode='classification'
    )

    # Explain one prediction (console-friendly)
    i = 5  # pick a test sample
    exp = explainer.explain_instance(
        data_row=X_test.iloc[i],
        predict_fn=pipeline.predict_proba
    )

    print(f"\nLIME explanation for test instance #{i}:")
    for feature, weight in exp.as_list():
        print(f"{feature}: {weight:.4f}")

if __name__ == "__main__":
    main()
