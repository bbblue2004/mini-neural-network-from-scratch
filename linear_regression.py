# Let's code a binary classifier from scratch with a logistic regression

import numpy as np
import matplotlib.pyplot as plt

###### DATASET ######

c0 = np.array([-2, -2])
c1 = np.array([2, 2])
sigma = 1.2 # standard deviation
n = 200 # number of points per class

### old code (not working)

# X0 = np.array([(c0 + np.random.normal(loc=0.0, scale=sigma, size=c1.shape), 0) for _ in range(200)])
# X1 = np.array([(c1 + np.random.normal(loc=0.0, scale=sigma, size=c2.shape), 1) for _ in range(200)])
# X = np.concatenate(X1, X2, axis=0)

### new code (correct)
np.random.seed(42)

def data():
    points_0 = c0 + np.random.normal(loc=0.0, scale=sigma, size=(n, 2))
    points_1 = c1 + np.random.normal(loc=0.0, scale=sigma, size=(n, 2))
    labels_0 = np.zeros(n)
    labels_1 = np.ones(n)

    X = np.concatenate([points_0, points_1], axis=0)   # np.concatenate(list of arrays, axis)
    y = np.concatenate([labels_0, labels_1], axis=0)

    return X, y

###### PLOT ######

def plot_data():
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap = 'viridis', edgecolors='k', alpha=0.7)   # plt.scatter permet de créer des nuages de points
    plt.title("Points du plan X et labels y = 0 ou 1.")
    plt.xlabel("Abscisse")
    plt.ylabel("Ordonnée")
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.savefig("data.png")
    plt.show()


###### Useful functions ######

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def accuracy(y, yhat):
    return np.sum(yhat == y) / len(y)

def binary_cross_entropy(y, yhat):
    yhat = np.clip(yhat, 1e-15, 1 - 1e-15)      # to avoid getting NaN
    return np.mean(-y * np.log(yhat) - (1 - y) * np.log(1 - yhat))

def forward():
    scores = np.array([sigmoid(np.dot(w, x) + b) for x in X])
    yhat = np.array([int(s >= 0.5) for s in scores])
    return yhat

###### Training of the model ######





if __name__ == '__main__':
    # data
    X, y = data()

    # parameters
    w = np.array([0, 0])
    b = 0

    yhat = forward()

    print(f"Initial accuracy: {accuracy(y, yhat)}")
    print(f"Initial loss: {binary_cross_entropy(y, yhat)}")