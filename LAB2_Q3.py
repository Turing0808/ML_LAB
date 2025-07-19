import pandas as pd
import matplotlib.pyplot as plt

# Load stock data
data = pd.read_excel(r"C:\Users\AKSHAT\OneDrive\Desktop\Machine Learning\LAB_2\Lab_Session_Data.xlsx", sheet_name='IRCTC Stock Price')

print("Mean stock price:", data['Price'].mean()) #mean
print("Variance in price:", data['Price'].var()) #variance

print("MP on Wednesdays:", data[data['Day'] == 'Wed']['Price'].mean()) #mean on wednesday
print("MP in April:", data[data['Month'] == 'Apr']['Price'].mean()) #mean on wednesday

print("PrP of loss:", (data['Chg%'] < 0).mean())
print("PrP of profit on Wednesday:", ((data['Day']=='Wed') & (data['Chg%'] > 0)).sum() / (data['Day']=='Wed').sum())

plt.scatter(data['Day'], data['Chg%']) # scatter plot using matlab plot library
plt.xlabel('Day')
plt.ylabel('Chg%')
plt.title('Change% by Day')
plt.show()
