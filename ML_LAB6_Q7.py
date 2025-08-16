import pandas as pd 
import numpy as np   
import matplotlib.pyplot as plt   
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder   

df = pd.read_csv(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\combine.csv")
df = df.dropna()   # Drop missing rows

target = "class1"   # Target column

# Convert features to numeric
for col in df.columns:
    if col != target:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()   # Drop rows with NaN

# Two features for decision boundary plot
feat1 = "duration"
feat2 = "flowPktsPerSecond"

X = df[[feat1, feat2]].values   # Features
y = df[target].values           # Target labels

# Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train decision tree
model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0)
model.fit(X, y_encoded)

# Define axis limits (zoomed in)
x_min, x_max = 0, 1e7   # Duration axis
y_min, y_max = 0, 1e5   # FlowPktsPerSecond axis

# Create mesh grid for plotting decision regions
xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Predict class for each grid point
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid_points)
Z = Z.astype(float).reshape(xx.shape)

# Plot decision boundary
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap="tab20")

# Plot actual data points
for cls in np.unique(y_encoded):
    plt.scatter(
        X[y_encoded == cls, 0], X[y_encoded == cls, 1],
        label=le.inverse_transform([cls])[0], edgecolor="k", s=40
    )

# Labels and legend
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel(feat1)
plt.ylabel(feat2)
plt.title("Decision Boundary with Zoomed Axes (Q7 - Option 2)")
plt.legend(title="Class")
plt.show()
