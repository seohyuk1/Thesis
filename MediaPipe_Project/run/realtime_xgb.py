import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import csv
import time

# =========================
# BASE PATH
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

# =========================
# MODEL PATH
# =========================
MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "xgb_model.pkl"
)

# =========================
# LOG PATH
# =========================
LOG_DIR = os.path.join(
    BASE_DIR,
    "logs",
    "xgb"
)

os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(
    LOG_DIR,
    "realtime_log.csv"
)

# =========================
# LABEL
# =========================
LABEL_MAP = {
    0: "squat",
    1: "pushup",
    2: "pullup"
}

# =========================
# LOAD MODEL
# =========================
if not os.path.exists(MODEL_PATH):

    print("모델 파일 없음:", MODEL_PATH)
    exit()

model = joblib.load(MODEL_PATH)

print("XGBoost 모델 로드 완료")

# =========================
# LOG FILE
# =========================
log_file = open(
    LOG_PATH,
    mode="w",
    newline="",
    encoding="utf-8"
)

log_writer = csv.writer(log_file)

log_writer.writerow([
    "timestamp",
    "action",
    "confidence",
    "fps"
])

print("로그 저장 위치:", LOG_PATH)

# =========================
# MEDIAPIPE
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

WINDOW_NAME = "Realtime Action Recognition (XGBoost)"

# =========================
# WINDOW
# =========================
cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

# fullscreen 제거
# 대신 큰 창으로 실행
cv2.resizeWindow(
    WINDOW_NAME,
    1600,
    900
)

# =========================
# FPS
# =========================
prev_time = 0

# =========================
# LOOP
# =========================
with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        # 좌우 반전
        image = cv2.cvtColor(
            cv2.flip(frame, 1),
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(image)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        action = "Waiting..."
        confidence = 0.0
        action_name = "none"

        # =========================
        # LANDMARK
        # =========================
        if results.pose_landmarks:

            try:
                landmarks = results.pose_landmarks.landmark

                row = []

                # =========================
                # 99 FEATURE
                # =========================
                for lm in landmarks:

                    row.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                # shape 안전 체크
                if len(row) == 99:

                    input_data = np.array(
                        row,
                        dtype=np.float32
                    ).reshape(1, -1)

                    # =========================
                    # PREDICT
                    # =========================
                    pred = model.predict(
                        input_data
                    )[0]

                    prob = model.predict_proba(
                        input_data
                    )[0]

                    confidence = float(
                        np.max(prob)
                    )

                    action_name = LABEL_MAP.get(
                        pred,
                        "unknown"
                    )

                    action = (
                        f"{action_name} "
                        f"({confidence:.2f})"
                    )

                # =========================
                # DRAW
                # =========================
                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            except Exception as e:

                print("에러:", e)

        # =========================
        # FPS
        # =========================
        curr_time = cv2.getTickCount()

        fps = (
            cv2.getTickFrequency() /
            (curr_time - prev_time)
            if prev_time != 0 else 0
        )

        prev_time = curr_time

        # =========================
        # LOG SAVE
        # =========================
        if action != "Waiting...":

            log_writer.writerow([
                time.time(),
                action_name,
                round(confidence, 4),
                round(fps, 2)
            ])

        # =========================
        # UI
        # =========================
        cv2.rectangle(
            image,
            (0, 0),
            (520, 110),
            (245, 117, 16),
            -1
        )

        cv2.putText(
            image,
            f"Action: {action}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 0, 0),
            2
        )

        cv2.putText(
            image,
            f"FPS: {int(fps)}",
            (10, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        # =========================
        # SHOW
        # =========================
        cv2.imshow(
            WINDOW_NAME,
            image
        )

        # 종료 키
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break

# =========================
# EXIT
# =========================
cap.release()

log_file.close()

cv2.destroyAllWindows()

print("로그 저장 완료")