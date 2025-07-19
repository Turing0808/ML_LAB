import pandas as pd
import numpy as np

df = pd.read_excel(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\LAB_2\Lab_Session_Data.xlsx", sheet_name="thyroid0387_UCI")

b_df = df.select_dtypes(include='number') # keep binary 

b_df = b_df.fillna(0) # Nan to 0 
b_df = (b_df != 0).astype(int) # Non zero to 1

v1 = b_df.iloc[0].values
v2 = b_df.iloc[1].values

f11 = np.sum((v1 == 1) & (v2 == 1))
f00 = np.sum((v1 == 0) & (v2 == 0))
f10 = np.sum((v1 == 1) & (v2 == 0))
f01 = np.sum((v1 == 0) & (v2 == 1))

# Safe division
jc_denom = f11 + f10 + f01
jc = f11 / jc_denom if jc_denom != 0 else 0

smc_denom = f11 + f10 + f01 + f00
smc = (f11 + f00) / smc_denom if smc_denom != 0 else 0

print("Jaccard Coefficient:", jc)
print("Simple Matching Coefficient:", smc)
