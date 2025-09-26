import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report

def load_data(filepath):
    # Load dataset, drop missing targets, remove messy 'duration' column, sample 20k rows
    data = pd.read_csv(filepath, low_memory=False)
    data = data.dropna(subset=['class1'])
    if 'duration' in data.columns:
        data = data.drop(columns=['duration'])
    data = data.sample(n=20000, random_state=42)
    return data

def preprocess_data(data):
    # Split into features (X) and target (y), then train/test sets
    X = data.drop(columns=['class1'])
    y = LabelEncoder().fit_transform(data['class1'])
    return train_test_split(X, y, test_size=0.2, random_state=42)

def build_stacking_classifier():
    # Build stacking classifier: Decision Tree + Random Forest → Logistic Regression meta model
    base_models = [
        ('dt', DecisionTreeClassifier(max_depth=10, random_state=42)),
        ('rf', RandomForestClassifier(n_estimators=30, max_depth=10, random_state=42))
    ]
    meta_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    stacking_clf = StackingClassifier(estimators=base_models, final_estimator=meta_model)
    
    # Pipeline: fill missing values → scale features → train model
    return make_pipeline(
        SimpleImputer(strategy='mean'),
        StandardScaler(),
        stacking_clf
    )

def main():
    # File path
    filepath = r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\cleaned.csv"

    # Load and prepare data
    data = load_data(filepath)
    X_train, X_test, y_train, y_test = preprocess_data(data)

    # Build and train model
    model = build_stacking_classifier()
    print(" Training model on 20k rows...")
    model.fit(X_train, y_train)

    # Evaluate model
    preds = model.predict(X_test)
    print("\n Accuracy:", accuracy_score(y_test, preds))
    print("\n Classification Report:\n", classification_report(y_test, preds))

if __name__ == "__main__":
    main()
