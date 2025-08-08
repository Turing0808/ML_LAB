import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = pd.read_csv("C:/Users/AKSHAT/OneDrive/Desktop/Machine Learning/ML_LAB_3/merged_vpn_data.csv", header=None) # Loading the dataset

numeric_cols = data.select_dtypes(include='number').columns # Picking one numeric feature and one target column
target_col = numeric_cols[0]     # Target variable
feature_col = numeric_cols[1]    # Single feature

X = data[[feature_col]]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Train and test split

model = LinearRegression().fit(X_train, y_train) # Training the model

y_train_pred = model.predict(X_train) # Predictions
y_test_pred = model.predict(X_test) # Predictions

print("Training Predictions:", y_train_pred[:5])
print("Test Predictions:", y_test_pred[:5])
