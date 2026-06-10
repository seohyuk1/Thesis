import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "landmark")

SAVE_X_PATH = os.path.join(BASE_DIR, "data", "dataset", "gnn", "X.npy")
SAVE_Y_PATH = os.path.join(BASE_DIR, "data", "dataset", "gnn", "y.npy")

LABEL_MAP = {
    "side_lunge": 0,
    "burpee": 1,
    "plank": 2,
    "pushup": 3,
    "crunch": 4
}

# =========================
# Normalize
# =========================
def normalize_landmarks(landmarks):

    # hip center 기준 정규화 (중요)
    lh = landmarks[23]
    rh = landmarks[24]
    hip_center = (lh + rh) / 2

    landmarks = landmarks - hip_center

    # scale normalization
    scale = np.max(np.linalg.norm(landmarks, axis=1)) + 1e-6
    landmarks = landmarks / scale

    return landmarks


def build_dataset():

    X, y = [], []

    print("\n===== GNN Dataset Build =====")

    files = os.listdir(DATA_PATH)

    for file in files:

        if not file.endswith(".csv"):
            continue

        label_name = None
        for label in LABEL_MAP.keys():
            if file.startswith(label):
                label_name = label
                break

        if label_name is None:
            continue

        label = LABEL_MAP[label_name]

        df = pd.read_csv(os.path.join(DATA_PATH, file))

        if len(df) < 5:
            continue

        data = df.drop(columns=["time", "mode"], errors="ignore")
        raw = data.values.astype(np.float32)

        # =========================
        # FRAME → SAMPLE (중요 변경)
        # =========================
        sequence = []

        for row in raw:

            if len(row) != 99:
                continue

            landmarks = np.array(row).reshape(33, 3)
            landmarks = normalize_landmarks(landmarks)

            sequence.append(landmarks)

        # 최소 프레임 필터
        if len(sequence) < 5:
            continue

        # 평균 representation (GNN baseline)
        sequence = np.mean(sequence, axis=0)  # (33, 3)

        X.append(sequence)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    os.makedirs(os.path.dirname(SAVE_X_PATH), exist_ok=True)

    np.save(SAVE_X_PATH, X)
    np.save(SAVE_Y_PATH, y)

    print("GNN dataset saved")


if __name__ == "__main__":
    build_dataset()