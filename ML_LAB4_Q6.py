import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv("C:/Users/AKSHAT/OneDrive/Desktop/Machine Learning/ML_LAB_3/merged_vpn_data.csv", header=None) # Loading dataset
df.columns = [f"F{i}" for i in range(len(df.columns)-1)] + ["Label"]

X = df[['F0', 'F1']].values # Select 2 features and the label
y = df['Label'].values

le = LabelEncoder() # Convert string labels to 0/1
y = le.fit_transform(y)  # VPN/Non-VPN → 0/1

X_train, _, y_train, _ = train_test_split(X, y, train_size=20, stratify=y, random_state=42) # Use 20 training samples (stratified)

k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X_train, y_train)

x_min, x_max = 0, 10 # Generate a safe grid (fixed range)
y_min, y_max = 0, 10
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
test_points = np.c_[xx.ravel(), yy.ravel()]

y_pred = model.predict(test_points) # Predict class of each test point

plt.figure(figsize=(7, 6)) # Plotting the decision region
plt.contourf(xx, yy, y_pred.reshape(xx.shape), cmap=plt.cm.RdBu, alpha=0.4)

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolor='k', s=100)
plt.xlabel("Feature F0")
plt.ylabel("Feature F1")
plt.title(f"kNN Decision Boundary using Project Data (k = {k})")
plt.grid(True)
plt.show()
