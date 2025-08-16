import pandas as pd   
import matplotlib.pyplot as plt   
from sklearn.tree import DecisionTreeClassifier, plot_tree   

df = pd.read_csv(r"C:\\Users\\AKSHAT\\OneDrive\\Desktop\\Machine Learning\\ML_LAB_6\\combine.csv")
df = df.dropna()   # Drop missing rows

target = "class1"   # Target column

# Convert all non-target columns to numeric
for col in df.columns:
    if col != target:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()   # Drop rows with NaN after conversion

# Split features and labels
X = df.drop(columns=[target])   # Features
y = df[target]                  # Labels

# Train decision tree classifier
model = DecisionTreeClassifier(criterion="entropy", max_depth=4, random_state=0)
model.fit(X, y)

# Visualize tree
plt.figure(figsize=(18, 10))   # Set figure size
plot_tree(
    model,
    feature_names=X.columns,   # Show feature names
    class_names=y.unique(),    # Show class labels
    filled=True,               # Color nodes
    rounded=True,              # Rounded boxes
    fontsize=9                 # Text size
)
plt.title("Decision Tree Visualization (Q6)")   
plt.show()   #
