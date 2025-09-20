# ML_LAB8_Q2.py
import numpy as np
import matplotlib.pyplot as plt
from ML_LAB8_Q1 import summation_unit, step_activation

def train_perceptron(inputs, labels, activation_fn,
                     lr=0.05, max_epochs=1000, tolerance=0.002,
                     init_weights=None, init_bias=0):
    """
    General perceptron trainer for N input features.
    - inputs: numpy array (N_samples x N_features)
    - labels: target outputs
    - activation_fn: activation function
    - lr: learning rate
    - init_weights: optional numpy array of starting weights
    - init_bias: starting bias
    """
    n_features = inputs.shape[1]
    if init_weights is None:
        weights = np.random.randn(n_features) * 0.1  # small random weights
    else:
        weights = np.array(init_weights, dtype=float)
    bias = init_bias
    errors_per_epoch = []

    for epoch in range(max_epochs):
        total_error = 0
        for x, y_true in zip(inputs, labels):
            summation = summation_unit(x, weights, bias)
            y_pred = activation_fn(summation)
            error = y_true - y_pred
            weights += lr * error * np.array(x)
            bias += lr * error
            total_error += error ** 2
        errors_per_epoch.append(total_error)
        if total_error <= tolerance:
            break
    return weights, bias, errors_per_epoch

# Example: AND Gate
if __name__ == "__main__":
    inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
    labels = np.array([0,0,0,1])
    w, b, errors = train_perceptron(inputs, labels, step_activation,
                                    lr=0.05, init_weights=[0.2,-0.75], init_bias=10)
    print("Final Weights:", w, "Bias:", b)
    import matplotlib.pyplot as plt
    plt.plot(errors)
    plt.xlabel("Epochs"); plt.ylabel("Error"); plt.title("AND Gate Training")
    plt.show()
