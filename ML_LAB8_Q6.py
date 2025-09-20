# ML_LAB8_Q6.py
import numpy as np
from ML_LAB8_Q1 import sigmoid_activation
from ML_LAB8_Q2 import train_perceptron

customer_inputs = np.array([
    [20,6,2],[16,3,6],[27,6,2],[19,1,2],[24,4,2],
    [22,1,5],[15,4,2],[18,4,2],[21,1,4],[16,2,4]
])
labels = np.array([1,1,1,0,1,0,1,1,0,0])  # High Value Tx?

w, b, errors = train_perceptron(customer_inputs, labels, sigmoid_activation,
                                lr=0.1, init_weights=[0.1,0.1,0.1], init_bias=0)
print("Customer Model Weights:", w, "Bias:", b)
