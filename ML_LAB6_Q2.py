import pandas as pd   
import numpy as np    

df = pd.read_csv(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\combine.csv")  
df = df.dropna()   # Drop rows with NaN
df.columns = df.columns.str.strip().str.lower()   # Clean column names

target = "class1"   # Target column

# Gini Index function
def gini_index(column):
    values, counts = np.unique(column, return_counts=True)   # Unique values & counts
    total = len(column)   # Total rows
    gini = 1
    for c in counts:
        p = c / total   # Probability of class
        gini -= p ** 2   # Subtract squared prob
    return gini   # Return Gini value

gini_val = gini_index(df[target])   # Compute Gini index of target
print("Gini Index of class:", gini_val)   # Print result
