import pandas as pd   
import numpy as np    
import math           

df = pd.read_csv(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\combine.csv")  # Load CSV
df.columns = df.columns.str.strip().str.lower()   # Clean column names
df = df.dropna()   # Drop missing rows

target = "class1"   # Target column

def entropy(column):  
    values, counts = np.unique(column, return_counts=True)   # Unique values & counts
    total = len(column)   # Total entries
    ent = 0
    for c in counts:  
        p = c / total   # Probability
        ent += -p * math.log2(p)   # Entropy formula
    return ent

print("Entropy of class:", entropy(df[target]))   
