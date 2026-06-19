from src.data import gen_inference, gen_train
from src.logistic_regression import LogisticRegression
from src.visualization import plot_losses, plot_train, plot_inference
from src.config import LR, EPOCHS


X_train, y_train = gen_train()
model = LogisticRegression(X_train, y_train, lr=LR)
plot_train(X_train, y_train)

model.train(epochs=EPOCHS, save_plot_losses=True)

X_inf = gen_inference()
predictions = model.inference(X_inf)
plot_inference(X_inf, predictions)