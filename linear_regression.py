# Let's code a binary classifier from scratch with a linear regression

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

points_0 = c0 + np.random.normal(loc=0.0, scale=sigma, size=(n, 2))
points_1 = c1 + np.random.normal(loc=0.0, scale=sigma, size=(n, 2))
labels_0 = np.zeros(n)
labels_1 = np.ones(n)

X = np.concatenate([points_0, points_1], axis=0)   # np.concatenate(list of arrays, axis)
y = np.concatenate([labels_0, labels_1], axis=0)


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


######