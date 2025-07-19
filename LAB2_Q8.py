import pandas as pd

def impute_simple(df):
    df_filled = df.copy()
    for col in df_filled.select_dtypes(include='number'):
        if df_filled[col].isnull().any():
            df_filled[col].fillna(df_filled[col].median(), inplace=True)
    for col in df_filled.select_dtypes(include='object'):
        if df_filled[col].isnull().any():
            df_filled[col].fillna(df_filled[col].mode()[0], inplace=True)
    return df_filled

if __name__ == "__main__":
    df = pd.read_excel(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\LAB_2\Lab_Session_Data.xlsx", sheet_name='thyroid0387_UCI')
    filled = impute_simple(df)
    print("Missing after imputation:")
    print(filled.isnull().sum())
