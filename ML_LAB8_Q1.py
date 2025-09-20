# ML_LAB8_Q1.py
import numpy as np

# Summation Unit
def summation_unit(inputs, weights, bias):
    return np.dot(inputs, weights) + bias

# Activation Functions
def step_activation(x): return 1 if x >= 0 else 0
def bipolar_step_activation(x): return 1 if x >= 0 else -1
def sigmoid_activation(x): return 1 / (1 + np.exp(-x))
def tanh_activation(x): return np.tanh(x)
def relu_activation(x): return max(0, x)
def leaky_relu_activation(x, alpha=0.01): return x if x > 0 else alpha * x

# Error Comparator
def error_comparator(y_true, y_pred):
    return (y_true - y_pred) ** 2

print(" A1 Functions Defined Successfully")
