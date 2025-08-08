import pandas as pd
from sklearn.cluster import KMeans

data = pd.read_csv("C:/Users/AKSHAT/OneDrive/Desktop/Machine Learning/ML_LAB_3/merged_vpn_data.csv", header=None) # Load the dataset

numeric_cols = data.select_dtypes(include='number').columns # Using all the numeric columns except first column 
X = data[numeric_cols.drop(numeric_cols[0])]

kmeans = KMeans(n_clusters=2, random_state=42, n_init="auto").fit(X) # K-Means

print("Cluster Labels:", kmeans.labels_[:10])
print("Cluster Centers:", kmeans.cluster_centers_)
