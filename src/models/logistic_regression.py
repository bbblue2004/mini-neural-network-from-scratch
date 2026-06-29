# Binary classifier from scratch with a logistic regression

import numpy as np
from src.utils.visualization import plot_losses
from src.utils.utils import sigmoid, binary_cross_entropy


###### MODEL #######


class LogisticRegression:

    def __init__(self, input_size, lr=0.1):
        self.w, self.b = np.zeros(input_size), 0.0
        self.lr = lr

    def predict(self, yhat):
        return yhat >= 0.5

    def accuracy(self, y, yhat):
        return np.mean((yhat >= 0.5) == y)

    def loss(self, y, yhat):         # binary cross-entropy
        yhat = np.clip(yhat, 1e-15, 1 - 1e-15)      # to avoid getting NaN
        return binary_cross_entropy(y, yhat)

    def forward(self, X):
        yhat = sigmoid(X @ self.w + self.b)   # or np.dot(w, x)
        return yhat

    def backprop(self, X, y, yhat):
        n = len(X)
        w_grad = np.dot(X.T, (yhat - y)) / n
        b_grad = np.ones(X.shape[0]).T @ (yhat - y) / n  # or np.mean(yhat - y)
        self.w -= self.lr * w_grad
        self.b -= self.lr * b_grad

    
    def train(self, X, y, epochs=10, loss_dir=None):
        losses = []

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs}\n")
            yhat = self.forward(X)
            self.backprop(X, y, yhat)
            current_loss = self.loss(y, yhat)
            losses.append(current_loss)
            print(f"Loss = {current_loss}")
            print(f"Accuracy = {self.accuracy(y, yhat)}\n")
        
        if loss_dir is not None:
            plot_losses(losses, loss_dir)

    
    def test(self, X_test, y_test):
        yhat_test = self.forward(X_test)
        predictions = self.predict(yhat_test)
        accuracy = self.accuracy(y_test, yhat_test)
        print(f"Test accuracy: {self.accuracy(y_test, yhat_test)}\n")
        return predictions

