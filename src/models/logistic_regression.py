# Binary classifier from scratch with a logistic regression

import numpy as np
from src.utils.visualization import plot_losses


def sigmoid(x):
    return 1 / (1 + np.exp(-x))



###### MODEL #######


class LogisticRegression:

    def __init__(self, X, y, lr=0.1):
        self.w, self.b = np.zeros(X.shape[1]), 0.0
        self.X, self.y = X, y
        self.lr = lr
        self.n = len(X)

    def predict(self, yhat):
        return yhat >= 0.5

    def accuracy(self, y, yhat):
        return np.mean((yhat >= 0.5) == y)

    def loss(self):         # binary cross-entropy
        # self.yhat = np.clip(self.yhat, 1e-15, 1 - 1e-15)      # to avoid getting NaN
        return np.mean(-self.y * np.log(self.yhat) - (1 - self.y) * np.log(1 - self.yhat))

    def forward(self, X):
        self.yhat = sigmoid(X @ self.w + self.b)   # or np.dot(w, x)
        return self.yhat

    def backprop(self):
        w_grad = np.dot(self.X.transpose(), (self.yhat - self.y)) / self.n
        b_grad = np.dot(np.ones(self.n).transpose(), (self.yhat - self.y)) / self.n  # or np.mean(self.yhat - self.y)
        self.w -= self.lr * w_grad
        self.b -= self.lr * b_grad

    
    def train(self, epochs=10, save_plot_losses=False):
        losses = []

        for epoch in range(1, epochs + 1):
            print(f"======= For epoch {epoch} =======\n")
            self.forward(self.X)
            current_loss = self.loss()
            losses.append(current_loss)
            self.backprop()
            print(f"Loss = {current_loss}")
            print(f"Accuracy = {self.accuracy(self.y, self.yhat)}\n")
        
        if save_plot_losses:
            plot_losses(losses)

    
    def inference(self, X_inf):
        yhat_inf = self.forward(X_inf)
        predictions = self.predict(yhat_inf)
        return predictions

