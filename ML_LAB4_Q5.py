import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def generate_training_data(): # Function to generate 20 training points
    np.random.seed(0)
    X_train = np.random.uniform(1, 10, size=(20, 2))
    y_train = np.array([0]*10 + [1]*10)
    return X_train, y_train

def generate_test_data(): # Function to create test grid
    x = np.arange(0, 10.1, 0.1)
    y = np.arange(0, 10.1, 0.1)
    xx, yy = np.meshgrid(x, y)
    test_points = np.c_[xx.ravel(), yy.ravel()]
    return test_points, xx, yy

def plot_for_different_k(X_train, y_train, test_points, xx, yy, k_values): # Function to plot kNN classification for different k
    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k) # Training kNN
        model.fit(X_train, y_train)

        y_pred = model.predict(test_points)  # Predict on test points

        plt.figure(figsize=(7, 6))
        plt.contourf(xx, yy, y_pred.reshape(xx.shape), alpha=0.4, cmap=plt.cm.RdBu)
        plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolor='k', s=100, label='Training Data')
        plt.title(f"kNN Classification with k = {k}")
        plt.xlabel("Feature X")
        plt.ylabel("Feature Y")
        plt.grid(True)
        plt.legend()
        plt.show()

if __name__ == "__main__":
    X_train, y_train = generate_training_data()
    test_points, xx, yy = generate_test_data()
    k_values = [1, 3, 5, 10]  # Try different k
    plot_for_different_k(X_train, y_train, test_points, xx, yy, k_values)
