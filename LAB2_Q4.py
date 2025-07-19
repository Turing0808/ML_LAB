import pandas as pd

df = pd.read_excel(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\LAB_2\Lab_Session_Data.xlsx", sheet_name="thyroid0387_UCI")

print("Column types:")
print(df.dtypes) # Datatypes

print("Missing values:")
print(df.isnull().sum())

print("Statistics of numeric columns:")
print(df.describe()) # Num col range

for col in df.select_dtypes(include='number').columns:
    print(f"{col}: Mean = {df[col].mean()}, Std = {df[col].std()}") # mean and std
