import numpy as np
from src.utils.utils import sigmoid, binary_cross_entropy
from src.utils.visualization import plot_losses
    

class Layer:

    def __init__(self, k, input_size, activation):
        self.k = k              # number of neurons in the layer
        self.W = np.random.randn(k, input_size) * np.sqrt(2.0 / input_size)       # at first all weights were set to 0 but the network could not learn
        self.b = np.zeros(k)
        self.activation = activation
    
    def forward(self, X):
        Z = X @ self.W.T + self.b      # actually, the formula z = W.x + b changes here because of vectorization: column vector x becomes line vector x.T!
        # self.b is automatically cast into a matrix
        return Z, self.activation(Z)

    # no backprop here, it's a small neural network, but it would be needed for a larger one


class NN():

    def __init__(self, input_size, hidden_size, lr=0.1):
        self.layer1 = Layer(hidden_size, input_size, np.tanh)    # 1st layer has 8 neurons of input_size 2
        self.layer2 = Layer(1, hidden_size, sigmoid)    # 2nd layer has 1 neuron of input_size 8
        self.lr = lr
        self.cache = {}                        # we add a cache to avoid recomputing intermediate values for backpropagation. Once during forward pass is enough.
    
    def accuracy(self, y, yhat):
        return np.mean((yhat >= 0.5) == y.reshape(yhat.shape))

    def predict(self, yhat):
        return yhat >= 0.5


    def forward(self, X):
        Z1, X1 = self.layer1.forward(X)
        Z2, X2 = self.layer2.forward(X1)
        self.cache["Z1"] = Z1
        self.cache["X1"] = X1
        return X2

    
    def loss(self, y, yhat):         # binary cross-entropy too
        yhat = np.clip(yhat, 1e-15, 1 - 1e-15)      # to avoid getting NaN
        return binary_cross_entropy(y.reshape(yhat.shape), yhat)

    def backprop(self, X, y, yhat):
            n = len(X)
            Z1 = self.cache["Z1"]
            X1 = self.cache["X1"]

            # to compute all those gradients: first do it for x a vector or R2, with a bit of differential calculus, the chain rule and the trick of dL = tr((grad).T dvariable)
            # then start again but with X of shape (n, 2), so with the new version of z(x)

            delta2 = yhat - y.reshape(yhat.shape)
            W2_grad = delta2.T @ X1 / n      
            b2_grad = np.mean(delta2)

            W1_grad = (1 / n) * ((delta2 @ self.layer2.W) * (1 - X1**2)).T @ X 
            b1_grad = np.mean(((delta2 @ self.layer2.W) * (1 - X1**2)), axis=0)

            self.layer2.W -= self.lr * W2_grad
            self.layer2.b -= self.lr * b2_grad
            self.layer1.W -= self.lr * W1_grad
            self.layer1.b -= self.lr * b1_grad

    
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
