import pandas as pd
import numpy as np
import os

# =========================
# 라벨 매핑
# =========================
LABEL_MAP = {
    "squat": 0,
    "pushup": 1,
    "pullup": 2
}

# =========================
# CSV 로드
# =========================
def load_csv_folder(folder_path):

    data = []

    for file in os.listdir(folder_path):

        if file.endswith(".csv"):

            df = pd.read_csv(
                os.path.join(folder_path, file)
            )

            data.append(df)

    if len(data) == 0:
        return pd.DataFrame()

    return pd.concat(data, ignore_index=True)


# =========================
# 라벨 인코딩
# =========================
def encode_label(label):

    return LABEL_MAP.get(label, -1)


def extract_labels(df):

    if "mode" not in df.columns:
        raise ValueError("mode column not found in dataframe")

    return df["mode"].apply(
        encode_label
    ).values


# =========================
# landmark → (33,3)
# =========================
def extract_landmarks(row):

    # time, mode 제거 (없어도 안전하게)
    data = row.drop(
        ["time", "mode"],
        errors="ignore"
    ).values

    data = np.array(data, dtype=np.float32)

    # =========================
    # 안전 체크 (핵심)
    # =========================
    if len(data) != 99:
        # 깨진 데이터 방지
        return np.zeros((33, 3), dtype=np.float32)

    return data.reshape(33, 3)


# =========================
# ST-GCN용 (T, V, C)
# =========================
def build_landmark_array(df):

    data = []

    for i in range(len(df)):

        row = df.iloc[i]

        skeleton = extract_landmarks(row)

        data.append(skeleton)

    return np.array(
        data,
        dtype=np.float32
    )


# =========================
# XGB / LSTM용 (99 feature)
# =========================
def extract_landmark_features(df):

    features = []

    for i in range(len(df)):

        row = df.iloc[i]

        lm = extract_landmarks(row)

        lm_flat = lm.flatten()

        # =========================
        # 안전장치 (99 고정)
        # =========================
        if lm_flat.shape[0] != 99:
            continue

        features.append(lm_flat)

    return np.array(
        features,
        dtype=np.float32
    )