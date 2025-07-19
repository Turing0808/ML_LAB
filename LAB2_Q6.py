import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

df = pd.read_excel(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\LAB_2\Lab_Session_Data.xlsx", sheet_name='thyroid0387_UCI')

for col in df.select_dtypes(include='object'):
    df[col+'_enc'] = LabelEncoder().fit_transform(df[col].astype(str))  # Encode all object columns

enc_columns = [c for c in df.columns if c.endswith('_enc')] # Encooded Cols for checking similarity
vec1 = df.loc[0, enc_columns].values.reshape(1, -1)
vec2 = df.loc[1, enc_columns].values.reshape(1, -1)

sim = cosine_similarity(vec1, vec2)[0][0]
print("Cosine similarity:", sim)
