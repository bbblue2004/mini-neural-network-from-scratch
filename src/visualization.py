import matplotlib.pyplot as plt
from src.config import IMAGES_DIR


def plot_losses(losses):
    plt.plot(range(len(losses)), losses)
    plt.savefig(IMAGES_DIR / "losses.png")
    # plt.show()


def plot_train(X, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap = 'viridis', edgecolors='k', alpha=0.7)   # plt.scatter permet de créer des nuages de points
    plt.title("Points du plan X et labels y = 0 ou 1.")
    plt.xlabel("Abscisse")
    plt.ylabel("Ordonnée")
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.savefig(IMAGES_DIR / "data.png")
    # plt.show()


def plot_inference(X_inf, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_inf[:, 0], X_inf[:, 1], c=predictions, cmap = 'coolwarm', edgecolors='k', alpha=0.7)   # plt.scatter permet de créer des nuages de points
    plt.title("Inference points with predicted label")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.savefig(IMAGES_DIR / "prediction.png")