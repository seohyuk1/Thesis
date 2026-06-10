import os
import numpy as np
import pandas as pd

# =========================
# BASE PATH
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "landmark")

SAVE_X_PATH = os.path.join(BASE_DIR, "data", "dataset", "xgb", "X.npy")
SAVE_Y_PATH = os.path.join(BASE_DIR, "data", "dataset", "xgb", "y.npy")

# =========================
# LABEL MAP
# =========================
LABEL_MAP = {
    "side_lunge": 0,
    "burpee": 1,
    "plank": 2,
    "pushup": 3,
    "crunch": 4
}

# =========================
# DISTANCE
# =========================
def calculate_distance(a, b):
    return np.linalg.norm(np.array(a) - np.array(b))

# =========================
# ANGLE
# =========================
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine = np.dot(ba, bc) / (
        np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6
    )

    cosine = np.clip(cosine, -1.0, 1.0)

    return np.degrees(np.arccos(cosine))

# =========================
# FEATURE EXTRACTOR
# =========================
def extract_features(row):

    # 안전 체크 추가
    if len(row) != 99:
        return None

    landmarks = row.reshape(33, 3)

    LS, RS = 11, 12
    LE, RE = 13, 14
    LW, RW = 15, 16
    LH, RH = 23, 24
    LK, RK = 25, 26
    LA, RA = 27, 28

    feature = []

    # =====================
    # ANGLES (12)
    # =====================
    feature.extend([
        calculate_angle(landmarks[LS], landmarks[LE], landmarks[LW]),
        calculate_angle(landmarks[RS], landmarks[RE], landmarks[RW]),
        calculate_angle(landmarks[LE], landmarks[LS], landmarks[LH]),
        calculate_angle(landmarks[RE], landmarks[RS], landmarks[RH]),
        calculate_angle(landmarks[LH], landmarks[LK], landmarks[LA]),
        calculate_angle(landmarks[RH], landmarks[RK], landmarks[RA]),
        calculate_angle(landmarks[LS], landmarks[LH], landmarks[LK]),
        calculate_angle(landmarks[RS], landmarks[RH], landmarks[RK]),
        calculate_angle(landmarks[LS], landmarks[LH], landmarks[RH]),
        calculate_angle(landmarks[RS], landmarks[RH], landmarks[LH]),
        calculate_angle(landmarks[LH], landmarks[LS], landmarks[RS]),
        calculate_angle(landmarks[RH], landmarks[RS], landmarks[LS]),
    ])

    # =====================
    # DISTANCES (10)
    # =====================
    feature.extend([
        calculate_distance(landmarks[LS], landmarks[RS]),
        calculate_distance(landmarks[LH], landmarks[RH]),
        calculate_distance(landmarks[LK], landmarks[RK]),
        calculate_distance(landmarks[LA], landmarks[RA]),
        calculate_distance(landmarks[LS], landmarks[LW]),
        calculate_distance(landmarks[RS], landmarks[RW]),
        calculate_distance(landmarks[LH], landmarks[LA]),
        calculate_distance(landmarks[RH], landmarks[RA]),
        calculate_distance(landmarks[LS], landmarks[LH]),
        calculate_distance(landmarks[RS], landmarks[RH]),
    ])

    # =====================
    # LEFT-RIGHT DIFFERENCE (4)
    # =====================
    feature.extend([
        abs(landmarks[LK][0] - landmarks[RK][0]),
        abs(landmarks[LA][0] - landmarks[RA][0]),
        abs(landmarks[LE][0] - landmarks[RE][0]),
        abs(landmarks[LW][0] - landmarks[RW][0]),
    ])

    # =====================
    # BODY TILT (2)
    # =====================
    shoulder_center = (landmarks[LS] + landmarks[RS]) / 2
    hip_center = (landmarks[LH] + landmarks[RH]) / 2

    feature.extend([
        shoulder_center[0] - hip_center[0],
        shoulder_center[1] - hip_center[1]
    ])

    return np.array(feature, dtype=np.float32)

# =========================
# DATASET BUILDER
# =========================
def build_dataset():

    X, y = [], []

    print("\n===== XGBoost Dataset Build =====")

    if not os.path.exists(DATA_PATH):
        print("데이터 폴더 없음")
        return

    files = os.listdir(DATA_PATH)

    for file in files:

        if not file.endswith(".csv"):
            continue

        file_path = os.path.join(DATA_PATH, file)

        label_name = None
        for label in LABEL_MAP.keys():
            if file.startswith(label):
                label_name = label
                break

        if label_name is None:
            print(f"라벨 인식 실패 : {file}")
            continue

        label = LABEL_MAP[label_name]

        try:
            df = pd.read_csv(file_path)
        except:
            continue

        data = df.drop(columns=["time", "mode"], errors="ignore")
        raw = data.values.astype(np.float32)

        count = 0

        for row in raw:

            feat = extract_features(row)

            # None 방지
            if feat is None:
                continue

            X.append(feat.astype(np.float32))
            y.append(label)

            count += 1

        print(f"[{label_name}] feature: {count}")

    X = np.array(X, dtype=np.float32)   # ⭐ 핵심 수정
    y = np.array(y, dtype=np.int32)

    print("\n===== RESULT =====")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    os.makedirs(os.path.dirname(SAVE_X_PATH), exist_ok=True)

    np.save(SAVE_X_PATH, X)
    np.save(SAVE_Y_PATH, y)

    print("\nDataset Saved")

# =========================
# RUN
# =========================
if __name__ == "__main__":
    build_dataset()