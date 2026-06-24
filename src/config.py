from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
IMAGES_DIR = ROOT_DIR / "images"

SEED = 42
LR = 0.1
EPOCHS = 300
SIGMA = 2
N_PER_CLASS = 200