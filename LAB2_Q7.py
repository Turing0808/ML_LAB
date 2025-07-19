import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_excel(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\LAB_2\Lab_Session_Data.xlsx", sheet_name="thyroid0387_UCI")
df = df.select_dtypes(include='number').fillna(0) #missing values filled with 0
data = df.iloc[:20].values

def cosine(a, b): #cosine similarity b/w 2 rows
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

sim_matrix = np.zeros((20, 20)) #empty similarity matrix
for i in range(20):
    for j in range(20):
        sim_matrix[i][j] = cosine(data[i], data[j])

sns.heatmap(sim_matrix, annot=False, cmap="viridis")
plt.title("Cosine Similarity Heatmap")
plt.show()
