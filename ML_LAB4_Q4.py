import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def generate_training_data(): # Function to generate training data 
    np.random.seed(0)
    X_train = np.random.uniform(1, 10, size=(20, 2))
    y_train = np.array([0]*10 + [1]*10)
    return X_train, y_train

def generate_test_data(): # Function to generate grid of 10,000 test points
    x = np.arange(0, 10.1, 0.1)
    y = np.arange(0, 10.1, 0.1)
    xx, yy = np.meshgrid(x, y)
    test_points = np.c_[xx.ravel(), yy.ravel()]
    return test_points, xx, yy

def classify_and_plot(X_train, y_train, test_points, xx, yy): # Function to classify and plot test points
    
    model = KNeighborsClassifier(n_neighbors=3) # Training kNN classifier with k=3
    model.fit(X_train, y_train)

    y_pred = model.predict(test_points) # Predict class of each test point

    plt.figure(figsize=(8, 6)) # Plot decision boundary 
    plt.contourf(xx, yy, y_pred.reshape(xx.shape), alpha=0.5, cmap=plt.cm.RdBu)

    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.RdBu, edgecolor='k', s=100, label='Training Data') # Plot training points

    plt.xlabel("Feature X")
    plt.ylabel("Feature Y")
    plt.title("Test Point Classification using kNN (k=3)")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    X_train, y_train = generate_training_data()
    test_points, xx, yy = generate_test_data()
    classify_and_plot(X_train, y_train, test_points, xx, yy)
