import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import os
from collections import deque
import csv
import time

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "lstm_model.keras"
)

# =========================
# LOG
# =========================
LOG_DIR = os.path.join(BASE_DIR, "logs", "lstm")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(
    LOG_DIR,
    "realtime_log.csv"
)

# =========================
# SETTINGS
# =========================
SEQ_LENGTH = 30

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

model = keras.models.load_model(MODEL_PATH)

print("LSTM 모델 로드 완료")

# =========================
# CSV LOG
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

# 해상도 설정
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 창 설정
WINDOW_NAME = "Realtime Action Recognition (LSTM)"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 1280, 720)

# =========================
# BUFFER
# =========================
sequence = deque(maxlen=SEQ_LENGTH)

# FPS 계산용
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

        # 좌우 반전 + RGB
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

        try:
            if results.pose_landmarks:

                landmarks = results.pose_landmarks.landmark

                row = []

                # (33,3) -> (99,)
                for lm in landmarks:
                    row.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                sequence.append(row)

                # =========================
                # PREDICTION
                # =========================
                if len(sequence) == SEQ_LENGTH:

                    input_data = np.array(sequence)

                    input_data = np.expand_dims(
                        input_data,
                        axis=0
                    )

                    pred = model.predict(
                        input_data,
                        verbose=0
                    )

                    pred_class = np.argmax(pred)

                    confidence = float(np.max(pred))

                    action_name = LABEL_MAP.get(
                        pred_class,
                        "unknown"
                    )

                    action = (
                        f"{action_name} "
                        f"({confidence:.2f})"
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
        # SAVE LOG
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
            (500, 110),
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
        # DRAW LANDMARK
        # =========================
        if results.pose_landmarks:

            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # =========================
        # SHOW
        # =========================
        cv2.imshow(WINDOW_NAME, image)

        # ESC 또는 q 종료
        key = cv2.waitKey(10)

        if key & 0xFF == ord('q') or key == 27:
            break

# =========================
# EXIT
# =========================
cap.release()

log_file.close()

cv2.destroyAllWindows()

print("로그 저장 완료")