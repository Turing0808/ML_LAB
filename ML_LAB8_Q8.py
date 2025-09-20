import numpy as np
from ML_LAB8_Q1 import sigmoid_activation

def sigmoid_derivative(x):
    return x * (1 - x)

# AND inputs and labels
inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
labels = np.array([0,0,0,1])

# Initialize weights
np.random.seed(1)
weights_input_hidden = np.random.rand(2,2)
weights_hidden_output = np.random.rand(2,1)
lr = 0.1
epochs = 2000

for epoch in range(epochs):
    # Forward pass
    hidden_input = np.dot(inputs, weights_input_hidden)
    hidden_output = sigmoid_activation(hidden_input)
    final_input = np.dot(hidden_output, weights_hidden_output)
    final_output = sigmoid_activation(final_input)

    # Compute error
    error = labels.reshape(-1,1) - final_output
    mse = np.mean(error**2)

    # Print progress every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, MSE: {mse:.4f}")

    # Backpropagation
    d_output = error * sigmoid_derivative(final_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Update weights
    weights_hidden_output += hidden_output.T.dot(d_output) * lr
    weights_input_hidden += inputs.T.dot(d_hidden) * lr

# Final outputs
print("\nTraining complete for AND gate")
print("Final outputs:")
print(final_output)
