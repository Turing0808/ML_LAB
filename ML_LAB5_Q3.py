import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score

data = pd.read_csv("C:/Users/AKSHAT/OneDrive/Desktop/Machine Learning/ML_LAB_3/merged_vpn_data.csv", header=None) # Load the dataset

numeric_cols = data.select_dtypes(include='number').columns # Select the numeric columns
target_col = numeric_cols[0]
X = data[numeric_cols.drop(target_col)]
y = data[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Train and test split

model = LinearRegression().fit(X_train, y_train) # Training the model

# Evaluate
def evaluate(model, X, y):
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    return mse, rmse, mape, r2

train_metrics = evaluate(model, X_train, y_train)
test_metrics = evaluate(model, X_test, y_test)

print("Train Metrics ->", train_metrics)
print("Test Metrics  ->", test_metrics)
