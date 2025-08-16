import pandas as pd   
import numpy as np    
import math           

df = pd.read_csv(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\combine.csv")
df = df.dropna()   # Drop missing rows
df.columns = df.columns.str.strip().str.lower()   # Clean column names

target = "class1"   # Target column

# Convert all non-target columns to numeric
for col in df.columns:
    if col != target:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()   # Drop rows with NaN after conversion

def entropy(column):
    values, counts = np.unique(column, return_counts=True)   # Unique values & counts
    total = len(column)
    ent = 0
    for c in counts:
        p = c / total   # Probability
        ent += -p * math.log2(p)   
    return ent

def information_gain(data, split_attr, target_attr):
    total_entropy = entropy(data[target_attr])   # Entropy of target
    values, counts = np.unique(data[split_attr], return_counts=True)
    weighted_entropy = 0
    for i in range(len(values)):
        subset = data[data[split_attr] == values[i]][target_attr]   # Subset of target
        weighted_entropy += (counts[i] / sum(counts)) * entropy(subset)   # Weighted entropy
    return total_entropy - weighted_entropy   # IG = parent entropy - weighted child entropy


def bin_feature(series, bins=4):
    return pd.qcut(series, q=bins, labels=False, duplicates="drop")   # Bin into quartiles


gains = {}
features_to_check = df.columns[:-1]   # All except target

for col in features_to_check:
    binned_col = bin_feature(df[col], bins=4)   # Bin feature into 4 intervals
    gains[col] = information_gain(df.assign(temp=binned_col), "temp", target)   # Compute IG

# Print results
print("Information Gain for each feature (qcut):")
for k, v in gains.items():
    print(f"{k}: {v}")

root = max(gains, key=gains.get)   # Feature with max IG
print("\nRoot node is:", root)
