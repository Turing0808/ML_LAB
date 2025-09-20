# ML_LAB8_Q10.py
import numpy as np

# Map: 0 → [1,0], 1 → [0,1]
labels_2out = np.array([[1,0],[0,1],[0,1],[1,0]])  # AND gate
print("Labels with 2 outputs:", labels_2out)
# Training would extend perceptron to 2 outputs (not implemented fully here).
