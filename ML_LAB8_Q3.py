# ML_LAB8_Q3.py
import numpy as np
from ML_LAB8_Q1 import step_activation, bipolar_step_activation, sigmoid_activation, relu_activation
from ML_LAB8_Q2 import train_perceptron

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
labels = np.array([0,0,0,1])  # AND gate

activations = [("Step", step_activation),
               ("Bi-Polar", bipolar_step_activation),
               ("Sigmoid", sigmoid_activation),
               ("ReLU", relu_activation)]

for name, fn in activations:
    _, _, errors = train_perceptron(inputs, labels, fn)
    print(f"{name} activation → Converged in {len(errors)} epochs")
