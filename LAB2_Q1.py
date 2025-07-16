import pandas as pd
import numpy as np

data = pd.read_excel("LAB2\Lab_Session_Data.xlsx",sheet_name="Purchase data")
print(data)

a=data[['Candies (#)','Mangoes (Kg)','Milk Packets (#)']].values
b=data[['Payment (Rs)']].values

Dimensionality=a.shape[1] # No of columns
vectors=b.shape[0] #No of rows

print(Dimensionality)
print(vectors)

rankmatrix=np.linalg.matrix_rank(a)

print(rankmatrix) # Rank of Matrix A

Inverse=np.linalg.pinv(a)
cost=np.dot(Inverse,b)
print(cost) # cost
