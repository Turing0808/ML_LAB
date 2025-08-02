import numpy as np
import matplotlib.pyplot as plt

def generate_and_plot_data(): # Function to generate and plot 2-class synthetic data
    np.random.seed(0)  # For reproducibility

    X = np.random.uniform(1, 10, size=(20, 2)) # Generating 20 points with X and Y between 1 and 10

    y = np.array([0]*10 + [1]*10) # Assign classes: first 10 are class 0 (Blue) and the next 10 are class 1 (Red)

    plt.figure(figsize=(6, 6))
    for i in range(len(X)):
        color = 'blue' if y[i] == 0 else 'red'
        plt.scatter(X[i, 0], X[i, 1], color=color, label=f'Class {y[i]}' if i in [0, 10] else "")

    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Synthetic 2D Training Data")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    generate_and_plot_data()
