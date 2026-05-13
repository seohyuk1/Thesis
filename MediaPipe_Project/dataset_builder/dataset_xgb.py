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
    "xgb",
    "X.npy"
)

SAVE_Y_PATH = os.path.join(
    BASE_DIR,
    "data",
    "dataset",
    "xgb",
    "y.npy"
)

# =========================
# LABEL
# =========================
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

    print("\n===== XGBoost 데이터 스캔 시작 =====")

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

        # =========================
        # FEATURE CHECK
        # =========================
        if data.shape[1] != 99:

            print("99 feature 아님 → skip")
            continue

        # =========================
        # FRAME 단위 저장
        # =========================
        count = 0

        for row in data:

            # shape 안정성
            if row.shape[0] != 99:
                continue

            X.append(row)
            y.append(label)

            count += 1

        print(f"추가된 프레임 수: {count}")

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
    # X shape: (5000, 99)

    os.makedirs(
        os.path.dirname(SAVE_X_PATH),
        exist_ok=True
    )

    np.save(SAVE_X_PATH, X)
    np.save(SAVE_Y_PATH, y)

    print("\nXGBoost Dataset saved!")

# =========================
# RUN
# =========================
if __name__ == "__main__":

    build_dataset()