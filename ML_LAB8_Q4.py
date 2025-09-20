# ML_LAB8_Q4.py
import numpy as np
import matplotlib.pyplot as plt
from ML_LAB8_Q1 import step_activation
from ML_LAB8_Q2 import train_perceptron

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
labels = np.array([0,0,0,1])

learning_rates = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
epochs_taken = []

for lr in learning_rates:
    _, _, errors = train_perceptron(inputs, labels, step_activation, lr=lr)
    epochs_taken.append(len(errors))

plt.plot(learning_rates, epochs_taken, marker='o')
plt.xlabel("Learning Rate"); plt.ylabel("Epochs to Converge")
plt.title("Learning Rate vs Epochs (AND Gate)")
plt.show()
