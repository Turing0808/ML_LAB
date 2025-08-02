import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def evaluate_knn_model(data, feature_columns, target_column, k=3): # Function for KNN model
    X = data[feature_columns]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) # Splitting data into train and test sets (70:30)

    model = KNeighborsClassifier(n_neighbors=k) # Training the KNN model
    model.fit(X_train, y_train)

    
    y_train_pred = model.predict(X_train) # Predicting for train sets
    y_test_pred = model.predict(X_test) # Predicting for test sets

    train_cm = confusion_matrix(y_train, y_train_pred) # Generating confusion matrix and classification report
    test_cm = confusion_matrix(y_test, y_test_pred)

    train_report = classification_report(y_train, y_train_pred, output_dict=True)
    test_report = classification_report(y_test, y_test_pred, output_dict=True)

    return train_cm, test_cm, train_report, test_report

if __name__ == "__main__":
    df = pd.read_csv("C:/Users/AKSHAT/OneDrive/Desktop/Machine Learning/ML_LAB_3/merged_vpn_data.csv", header=None)

    df.columns = [f"F{i}" for i in range(len(df.columns)-1)] + ["Label"]  # assuming that the last column is label

    print("Dataset Columns:", df.columns.tolist())

    selected_features = ['F0', 'F1', 'F2', 'F3'] # Select a few simple numeric features from F0, F1, F2, F3 (beginner style)
    label_column = 'Label'
    train_cm, test_cm, train_report, test_report = evaluate_knn_model(df, selected_features, label_column)

    print("\nTraining Confusion Matrix:\n", train_cm)
    print("\nTraining Classification Report:")
    print(pd.DataFrame(train_report).transpose())

    print("\nTesting Confusion Matrix:\n", test_cm)
    print("\nTesting Classification Report:")
    print(pd.DataFrame(test_report).transpose())
