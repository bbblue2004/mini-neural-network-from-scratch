import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def binary_cross_entropy(y, yhat):
    return np.mean(-y * np.log(yhat) - (1 - y) * np.log(1 - yhat))