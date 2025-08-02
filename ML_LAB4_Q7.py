import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV, train_test_split

df = pd.read_csv("C:/Users/AKSHAT/OneDrive/Desktop/Machine Learning/ML_LAB_3/merged_vpn_data.csv", header=None) # Loading dataset
df.columns = [f"F{i}" for i in range(len(df.columns)-1)] + ['Label']

X = df[['F0', 'F1']].values # Use 2 features and label
y = df['Label'].values

le = LabelEncoder() # Encode labels to 0/1
y = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42) # Train/test split

param_grid = {'n_neighbors': list(range(1, 21))} # Set up parameter grid

model = KNeighborsClassifier()
grid = GridSearchCV(model, param_grid, cv=5)  # 5-fold cross validation,Using GridSearchCV to find best k
grid.fit(X_train, y_train)

best_k = grid.best_params_['n_neighbors'] # Printing the best k 
test_accuracy = grid.score(X_test, y_test) # Printing the test accuracy for that best k

print("Best k value:", best_k)
print("Test Accuracy with best k:", test_accuracy)
