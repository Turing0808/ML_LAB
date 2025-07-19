import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from LAB2_Q8 import impute_simple

df = pd.read_excel(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\LAB_2\Lab_Session_Data.xlsx", sheet_name='thyroid0387_UCI')
df = impute_simple(df)

# Normalize numbers
num_cols = df.select_dtypes(include='number').columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print(df[num_cols].head())
