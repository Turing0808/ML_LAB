# ML_LAB8_Q7.py
import numpy as np

customer_inputs = np.array([
    [20,6,2],[16,3,6],[27,6,2],[19,1,2],[24,4,2],
    [22,1,5],[15,4,2],[18,4,2],[21,1,4],[16,2,4]
])
labels = np.array([1,1,1,0,1,0,1,1,0,0])

X = np.hstack([customer_inputs, np.ones((len(customer_inputs),1))]) # add bias
y = labels.reshape(-1,1)

w_pinv = np.linalg.pinv(X).dot(y)
print("Weights from Pseudo-Inverse:", w_pinv.flatten())
