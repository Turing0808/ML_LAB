import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

data = pd.read_csv("C:/Users/AKSHAT/OneDrive/Desktop/Machine Learning/ML_LAB_3/merged_vpn_data.csv", header=None) # Load the dataset

numeric_cols = data.select_dtypes(include='number').columns # Features for clustering
X = data[numeric_cols.drop(numeric_cols[0])]

# Trying multiple k values
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(X)
    sil = silhouette_score(X, kmeans.labels_)
    ch = calinski_harabasz_score(X, kmeans.labels_)
    db = davies_bouldin_score(X, kmeans.labels_)
    print(f"k={k} -> Silhouette: {sil:.4f}, CH: {ch:.2f}, DB: {db:.4f}")
