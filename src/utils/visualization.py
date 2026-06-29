import matplotlib.pyplot as plt
from src.config import IMAGES_DIR


def plot_losses(losses, dirname):
    out_dir = IMAGES_DIR / dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.title("Training loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(range(1, len(losses) + 1), losses)
    plt.savefig(out_dir / "losses.png")
    plt.close()


def plot_train(X, y, dirname):
    out_dir = IMAGES_DIR / dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap = 'viridis', edgecolors='k', alpha=0.7)
    plt.title("Training data (labels 0 and 1)")
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.savefig(out_dir / f"train_data.png")
    plt.close()


def plot_test(X_test, predictions, dirname):
    out_dir = IMAGES_DIR / dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.scatter(X_test[:, 0], X_test[:, 1], c=predictions, cmap = 'coolwarm', edgecolors='k', alpha=0.7)
    plt.title("Test points with predicted label")
    plt.xlabel("x-coordinate")
    plt.ylabel("y-coordinate")
    plt.grid(True, linestyle='--', alpha=0.8)
    plt.savefig(out_dir / f"test_data.png")
    plt.close()