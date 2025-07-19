import pandas as pd

customer_data = pd.read_excel(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\LAB_2\Lab_Session_Data.xlsx", sheet_name='Purchase data') # Loading Data

def label_customer(payment):
    return 'RICH' if payment > 200 else 'POOR'

customer_data['Class'] = customer_data['Payment (Rs)'].apply(label_customer)
print(customer_data[['Customer', 'Payment (Rs)', 'Class']])
