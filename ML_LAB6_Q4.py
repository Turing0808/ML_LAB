import pandas as pd  

df = pd.read_csv(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\ML_LAB_6\combine.csv")
df = df.dropna()   # Drop missing rows
df.columns = df.columns.str.strip().str.lower()   # Clean column names

target = "class1"   # Target column

# Convert all non-target columns to numeric
for col in df.columns:
    if col != target:
        df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna()   # Drop rows with NaN after conversion

def binning(series, bins=4, method="width"):
    if method == "width":
        return pd.cut(series, bins=bins, labels=False)   # Equal-width binning
    elif method == "frequency":
        return pd.qcut(series, q=bins, labels=False, duplicates="drop")   # Equal-frequency binning
    else:
        raise ValueError("Method must be 'width' or 'frequency'")

df["duration_width"] = binning(df["duration"], bins=4, method="width")     # Width-based binning
df["duration_freq"] = binning(df["duration"], bins=4, method="frequency") # Frequency-based binning


print(df[["duration", "duration_width", "duration_freq"]].head(10))
