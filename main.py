import numpy as np

from src.data import gen_test, gen_train, gen_non_lin_sep_train, gen_non_lin_sep_test
from src.models.logistic_regression import LogisticRegression
from src.models.neural_network import NN
from src.utils.visualization import plot_losses, plot_train, plot_test
from src.config import LR, EPOCHS, SEED


def run_logistic_regression():
    dirname = "log_reg"
    X_train, y_train = gen_train()
    input_size = X_train.shape[1]

    model = LogisticRegression(input_size, lr=LR)
    plot_train(X_train, y_train, dirname)
    model.train(X_train, y_train, epochs=EPOCHS, loss_dir=dirname)

    X_test, y_test = gen_test()
    predictions = model.test(X_test, y_test)
    plot_test(X_test, predictions, dirname)


def run_logistic_on_nonlinear():
    dirname = "log_reg_nonlin"
    X_train, y_train = gen_non_lin_sep_train()
    input_size = X_train.shape[1]

    model = LogisticRegression(input_size, lr=LR)
    plot_train(X_train, y_train, dirname)

    model.train(X_train, y_train, epochs=EPOCHS, loss_dir=dirname)

    X_test, y_test = gen_non_lin_sep_test()
    predictions = model.test(X_test, y_test)
    plot_test(X_test, predictions, dirname)   


def run_neural_network():
    dirname = "nn"
    X_train, y_train = gen_non_lin_sep_train()
    input_size = X_train.shape[1]

    model = NN(input_size, 8, lr=LR)
    plot_train(X_train, y_train, dirname)

    model.train(X_train, y_train, epochs=EPOCHS, loss_dir=dirname)

    X_test, y_test = gen_non_lin_sep_test()
    predictions = model.test(X_test, y_test)
    plot_test(X_test, predictions, dirname)


if __name__ == '__main__':
    np.random.seed(SEED)
    run_logistic_regression()
    run_logistic_on_nonlinear()
    run_neural_network()