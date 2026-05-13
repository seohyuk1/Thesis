import os
import numpy as np
import pandas as pd

# =========================
# BASE PATH
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "raw",
    "landmark"
)

SAVE_X_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "lstm",
    "X.npy"
)

SAVE_Y_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "lstm",
    "y.npy"
)

# =========================
# CONFIG
# =========================
SEQ_LENGTH = 30
STRIDE = 10

LABEL_MAP = {
    "squat": 0,
    "pushup": 1,
    "pullup": 2
}

# =========================
# DATASET BUILDER
# =========================
def build_dataset():

    X = []
    y = []

    print("\n===== LSTM 데이터 스캔 시작 =====")

    for file in os.listdir(DATA_PATH):

        print(f"\n파일 발견: {file}")

        # csv만 처리
        if not file.endswith(".csv"):
            print("→ csv 아님, skip")
            continue

        file_path = os.path.join(
            DATA_PATH,
            file
        )

        # =========================
        # LABEL
        # =========================
        label_name = os.path.splitext(file)[0].split("_")[0]

        print("라벨 추출:", label_name)

        if label_name not in LABEL_MAP:
            print("LABEL_MAP에 없음 → skip")
            continue

        label = LABEL_MAP[label_name]

        # =========================
        # CSV LOAD
        # =========================
        try:
            df = pd.read_csv(file_path)

        except Exception as e:
            print("로드 실패:", e)
            continue

        # 빈 데이터 방어
        if len(df) == 0:
            print("빈 파일 → skip")
            continue

        # =========================
        # 컬럼 제거
        # =========================
        try:
            data = df.drop(
                columns=["time", "mode"],
                errors="ignore"
            ).values.astype(np.float32)

        except Exception as e:
            print("데이터 변환 실패:", e)
            continue

        print("사용 데이터 shape:", data.shape)

        # 99 feature 체크
        if data.shape[1] != 99:
            print("feature 수 이상 → skip")
            continue

        # 프레임 부족
        if len(data) < SEQ_LENGTH:
            print("프레임 부족 → skip")
            continue

        # =========================
        # WINDOW SPLIT
        # =========================
        count = 0

        for start in range(
            0,
            len(data) - SEQ_LENGTH + 1,
            STRIDE
        ):

            seq = data[
                start:start + SEQ_LENGTH
            ]

            # shape 안정성
            if seq.shape != (SEQ_LENGTH, 99):
                continue

            X.append(seq)
            y.append(label)

            count += 1

        print(f"추가된 샘플 수: {count}")

    # =========================
    # FINAL
    # =========================
    if len(X) == 0:

        print("\n데이터가 생성되지 않았습니다!")
        return

    X = np.array(
        X,
        dtype=np.float32
    )

    y = np.array(
        y,
        dtype=np.int32
    )

    print("\n===== 최종 결과 =====")

    print("X shape:", X.shape)
    print("y shape:", y.shape)

    # 예:
    # X shape: (120, 30, 99)

    os.makedirs(
        os.path.dirname(SAVE_X_PATH),
        exist_ok=True
    )

    np.save(SAVE_X_PATH, X)
    np.save(SAVE_Y_PATH, y)

    print("\nLSTM Dataset saved!")

# =========================
# RUN
# =========================
if __name__ == "__main__":

    build_dataset()