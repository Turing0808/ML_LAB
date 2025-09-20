# ML_LAB8_Q11.py
import numpy as np
from sklearn.neural_network import MLPClassifier

inputs = np.array([[0,0],[0,1],[1,0],[1,1]])
labels_and = np.array([0,0,0,1])
labels_xor = np.array([0,1,1,0])

mlp_and = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
mlp_and.fit(inputs, labels_and)
print("AND Gate Prediction:", mlp_and.predict(inputs))

mlp_xor = MLPClassifier(hidden_layer_sizes=(4,), max_iter=2000)
mlp_xor.fit(inputs, labels_xor)
print("XOR Gate Prediction:", mlp_xor.predict(inputs))
