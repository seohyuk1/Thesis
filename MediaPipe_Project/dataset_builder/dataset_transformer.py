import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

X_PATH = os.path.join(BASE_DIR, "data", "dataset", "gnn", "X.npy")
Y_PATH = os.path.join(BASE_DIR, "data", "dataset", "gnn", "y.npy")

SAVE_X_PATH = os.path.join(BASE_DIR, "data", "dataset", "transformer", "X.npy")
SAVE_Y_PATH = os.path.join(BASE_DIR, "data", "dataset", "transformer", "y.npy")

def build_dataset():

    X = np.load(X_PATH)   # (N, 33, 3)
    y = np.load(Y_PATH)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    os.makedirs(os.path.dirname(SAVE_X_PATH), exist_ok=True)

    np.save(SAVE_X_PATH, X)
    np.save(SAVE_Y_PATH, y)

    print("Transformer dataset saved")

if __name__ == "__main__":
    build_dataset()