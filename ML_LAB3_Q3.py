import numpy as np
import matplotlib.pyplot as plt
from ML_LAB3_Q1 import get_data

def minkowski_distance(row1, row2):
    orders = range(1, 11)
    distances = []

    for r in orders:
        dist = np.sum(np.abs(row1 - row2) ** r) ** (1. / r)
        distances.append(dist)

    plt.plot(list(orders), distances, marker='o', color='darkorange') # plot distance vs r 
    plt.title("Minkowski Distance vs r")
    plt.xlabel("Order (r)")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

    return distances

features, _ = get_data()
minkowski_distance(features[0], features[1])
