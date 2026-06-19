# Binary classifier from scratch with a logistic regression

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(42)

###### DATASET ######

class Dataset:

    def __init__(self):
        self.X, self.y = self.gen()


    def gen(self):
        c0 = np.array([-2, -2])
        c1 = np.array([2, 2])
        sigma = 2 # standard deviation
        n = 200 # number of points per class
        
        points_0 = c0 + np.random.normal(loc=0.0, scale=sigma, size=(n, 2))
        points_1 = c1 + np.random.normal(loc=0.0, scale=sigma, size=(n, 2))
        labels_0 = np.zeros(n)
        labels_1 = np.ones(n)

        X = np.concatenate([points_0, points_1], axis=0)   # np.concatenate(list of arrays, axis)
        y = np.concatenate([labels_0, labels_1], axis=0)

        return X, y


    def plot(self):
        plt.figure(figsize=(8, 6))
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.y, cmap = 'viridis', edgecolors='k', alpha=0.7)   # plt.scatter permet de créer des nuages de points
        plt.title("Points du plan X et labels y = 0 ou 1.")
        plt.xlabel("Abscisse")
        plt.ylabel("Ordonnée")
        plt.grid(True, linestyle='--', alpha=0.8)
        plt.savefig("data.png")
        # plt.show()


###### A FEW FUNCTIONS ######

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def plot_losses(losses):
    plt.plot(range(len(losses)), losses)
    plt.savefig("losses")
    # plt.show()


###### MODEL #######


class LogisticRegression:

    def __init__(self, X, y, lr=0.1):
        self.w, self.b = np.array([0.0, 0]), 0.0
        self.X, self.y = X, y
        self.yhat = self.forward()
        self.lr = lr
        self.n = len(data.X)

    def accuracy(self):
        return np.mean((self.yhat >= 0.5) == self.y)

    def loss(self):         # binary cross-entropy
        # self.yhat = np.clip(self.yhat, 1e-15, 1 - 1e-15)      # to avoid getting NaN
        return np.mean(-self.y * np.log(self.yhat) - (1 - self.y) * np.log(1 - self.yhat))

    def forward(self):
        self.yhat = sigmoid(self.X @ self.w + self.b)   # or np.dot(w, x)
        return self.yhat

    def backprop(self):
        w_grad = np.dot(self.X.transpose(), (self.yhat - self.y)) / self.n
        b_grad = np.dot(np.ones(self.n).transpose(), (self.yhat - self.y)) / self.n  # or np.mean(self.yhat - self.y)
        self.w -= self.lr * w_grad
        self.b -= self.lr * b_grad

    


if __name__ == '__main__':

    data = Dataset()
    model = LogisticRegression(data.X, data.y)
    losses = []
    data.plot()

    for epoch in range(1, 10):
        print(f"======= For epoch {epoch} =======\n")
        model.forward()
        current_loss = model.loss()
        losses.append(current_loss)
        model.backprop()
        print(f"Loss = {current_loss}")
        print(f"Accuracy = {model.accuracy()}\n")

    
    plot_losses(losses)