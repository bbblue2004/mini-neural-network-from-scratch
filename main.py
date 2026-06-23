from src.data import gen_inference, gen_train, gen_non_lin_sep_train
from src.models.logistic_regression import LogisticRegression
from src.utils.visualization import plot_losses, plot_train, plot_inference, plot_non_lin_sep_data
from src.config import LR, EPOCHS
import sys


X_train, y_train = gen_non_lin_sep_train()
model = LogisticRegression(X_train, y_train, lr=LR)
plot_non_lin_sep_data(X_train, y_train)
sys.exit()

model.train(epochs=EPOCHS, save_plot_losses=True)

X_inf = gen_inference()
predictions = model.inference(X_inf)
plot_inference(X_inf, predictions)