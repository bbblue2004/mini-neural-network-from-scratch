import numpy as np

from src.data import gen_test, gen_train, gen_non_lin_sep_train, gen_non_lin_sep_test
from src.models.logistic_regression import LogisticRegression
from src.models.neural_network import NN
from src.utils.visualization import plot_losses, plot_train, plot_test, plot_non_lin_sep_train, plot_non_lin_sep_test
from src.config import LR, EPOCHS, SEED
import sys
np.random.seed(SEED)

X_train, y_train = gen_non_lin_sep_train()
input_size = X_train.shape[1]

model = NN(input_size, 8, lr=LR)
plot_non_lin_sep_train(X_train, y_train)

model.train(X_train, y_train, epochs=EPOCHS, save_plot_losses=True)

X_test, y_test = gen_non_lin_sep_test()
predictions = model.test(X_test, y_test)
plot_non_lin_sep_test(X_test, predictions)