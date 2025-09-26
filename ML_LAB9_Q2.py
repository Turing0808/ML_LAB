import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

def load_data(filepath):
    # Load dataset and remove rows where target 'class1' is missing
    data = pd.read_csv(filepath)
    data = data.dropna(subset=['class1'])
    return data

def preprocess_data(data):
    # Split into features (X) and target (y) and encode labels
    X = data.drop(columns=['class1'])
    y = LabelEncoder().fit_transform(data['class1'])
    
    # Split into train and test sets (80% train, 20% test)
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_pipeline():
    # Build a pipeline: fill missing values → scale features → train Random Forest
    return Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),   # Replace missing values with mean
        ('scaler', StandardScaler()),                  # Standardize features
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))  # Random Forest
    ])

def main():
    # Path to dataset
    filepath = r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\cleaned.csv"

    # Load and prepare data
    data = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Build pipeline and train model
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # Make predictions and evaluate accuracy
    preds = pipeline.predict(X_test)
    print("Pipeline Accuracy:", accuracy_score(y_test, preds))

if __name__ == "__main__":
    main()
