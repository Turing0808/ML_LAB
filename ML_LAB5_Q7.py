import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv("C:/Users/AKSHAT/OneDrive/Desktop/Machine Learning/ML_LAB_3/merged_vpn_data.csv", header=None) # Load the dataset

numeric_cols = data.select_dtypes(include='number').columns # Features for clustering
X = data[numeric_cols.drop(numeric_cols[0])]

# Elbow method
distortions = []
for k in range(2, 20):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    distortions.append(kmeans.inertia_)

plt.plot(range(2, 20), distortions, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Distortion (Inertia)")
plt.title("Elbow Method for Optimal k")
plt.show()
