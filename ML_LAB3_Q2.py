import matplotlib.pyplot as plt
import numpy as np
from ML_LAB3_Q1 import get_data  

def show_feature_plot(values):
    avg = np.mean(values) # mean of column
    var = np.var(values) # variance of column

    plt.hist(values, bins=20, color='slateblue', edgecolor='black')
    plt.title("Feature Distribution")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.show()

    return avg, var

features, labels = get_data()
chosen_column = features[:, 10]  # change index to visualize other columns
mean_value, var_value = show_feature_plot(chosen_column)
print("The Mean is : ", mean_value)
print("The Variance is : ", var_value)
