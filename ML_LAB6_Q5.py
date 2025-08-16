import pandas as pd   
import numpy as np    
import math           


df = pd.read_csv(r"C:\\Users\\AKSHAT\\OneDrive\\Desktop\\Machine Learning\\ML_LAB_6\\combine.csv")
df = df.dropna()   # Drop missing rows
df.columns = df.columns.str.strip().str.lower()   # Clean column names

target = "class1"   # Target column

# Convert features to numeric where possible
for col in df.columns:
    if col != target:
        df[col] = pd.to_numeric(df[col], errors="coerce")
df = df.dropna()   # Drop rows with NaN

def entropy(column):
    values, counts = np.unique(column, return_counts=True)
    total = len(column)
    return sum(- (c/total) * math.log2(c/total) for c in counts)   

def information_gain(data, split_attr, target_attr):
    total_ent = entropy(data[target_attr])   # Entropy of target
    vals, cnts = np.unique(data[split_attr], return_counts=True)
    weighted = sum((cnt/sum(cnts)) * entropy(data[data[split_attr] == v][target_attr])
                   for v, cnt in zip(vals, cnts))   # Weighted entropy
    return total_ent - weighted   # IG = parent - weighted child


def bin_feature(series, bins=4):
    return pd.qcut(series, q=bins, labels=False, duplicates="drop")

def find_best_feature(data):
    ig_scores = {}
    for col in data.columns:
        if col != target:
            try:
                binned = bin_feature(data[col])   # Bin feature
                ig_scores[col] = information_gain(data.assign(tmp=binned), "tmp", target)
            except Exception:
                continue
    return max(ig_scores, key=ig_scores.get)   # Feature with highest IG

def build_tree(data, depth=0, max_depth=3):
    print("  " * depth + f"Node depth {depth}, samples={len(data)}")

    # If all rows belong to one class → Leaf
    if len(np.unique(data[target])) == 1:
        result = np.unique(data[target])[0]
        print("  " * depth + f"Leaf → {result}")
        return result

    # Stop if max depth reached → majority class
    if depth >= max_depth:
        result = data[target].mode()[0]
        print("  " * depth + f"Max depth reached → majority={result}")
        return result

    # Find best splitting feature
    best_feature = find_best_feature(data)
    print("  " * depth + f"Best feature: {best_feature}")

    tree = {best_feature: {}}
    binned = bin_feature(data[best_feature])

    # Split for each bin value
    for val in np.unique(binned):
        subset = data[binned == val]
        if subset.empty:
            tree[best_feature][val] = data[target].mode()[0]   # Default to majority
        else:
            tree[best_feature][val] = build_tree(subset, depth+1, max_depth)
    return tree


tree = build_tree(df, max_depth=3)

print("\nFinal Decision Tree (limited depth):")
print(tree)
