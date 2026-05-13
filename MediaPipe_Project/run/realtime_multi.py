import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import joblib
import os
import time
import csv
from collections import deque

from ensemble import ensemble_predict

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
XGB_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "xgb_model.pkl"
)

LSTM_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "lstm_model.keras"
)

STGCN_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "stgcn_model.keras"
)

# =========================
# LOG
# =========================
LOG_DIR = os.path.join(
    BASE_DIR,
    "logs",
    "ensemble"
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
print("Loading models...")

xgb_model = joblib.load(
    XGB_MODEL_PATH
)

lstm_model = tf.keras.models.load_model(
    LSTM_MODEL_PATH
)

stgcn_model = tf.keras.models.load_model(
    STGCN_MODEL_PATH
)

print("All models loaded!")

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

# =========================
# CONFIG
# =========================
SEQ_LENGTH = 30

# =========================
# SEQUENCE BUFFER
# =========================
sequence = deque(
    maxlen=SEQ_LENGTH
)

# =========================
# MEDIAPIPE
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

WINDOW_NAME = "Realtime Multi Model Ensemble"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
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

        # =========================
        # LANDMARK
        # =========================
        if results.pose_landmarks:

            landmarks = results.pose_landmarks.landmark

            row = []

            for lm in landmarks:

                row.extend([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            row = np.array(
                row,
                dtype=np.float32
            )

            # =========================
            # XGB INPUT
            # =========================
            xgb_input = row.reshape(1, -1)

            xgb_prob = xgb_model.predict_proba(
                xgb_input
            )[0]

            # =========================
            # SEQUENCE APPEND
            # =========================
            sequence.append(row)

            # =========================
            # LSTM/STGCN
            # =========================
            if len(sequence) == SEQ_LENGTH:

                seq_data = np.array(
                    sequence,
                    dtype=np.float32
                )

                # =========================
                # LSTM INPUT
                # (1,30,99)
                # =========================
                lstm_input = seq_data.reshape(
                    1,
                    SEQ_LENGTH,
                    99
                )

                lstm_prob = lstm_model.predict(
                    lstm_input,
                    verbose=0
                )[0]

                # =========================
                # ST-GCN INPUT
                # (1,30,33,3)
                # =========================
                stgcn_input = seq_data.reshape(
                    1,
                    SEQ_LENGTH,
                    33,
                    3
                )

                stgcn_prob = stgcn_model.predict(
                    stgcn_input,
                    verbose=0
                )[0]

                # =========================
                # ENSEMBLE
                # =========================
                result = ensemble_predict(
                    xgb_prob,
                    lstm_prob,
                    stgcn_prob,
                    method="weighted"
                )

                action = result["action"]

                confidence = result["confidence"]

                action = (
                    f"{action} "
                    f"({confidence:.2f})"
                )

                # =========================
                # SAVE LOG
                # =========================
                curr_time_log = time.time()

                # FPS 계산
                curr_tick = cv2.getTickCount()

                fps = (
                    cv2.getTickFrequency() /
                    (curr_tick - prev_time)
                    if prev_time != 0 else 0
                )

                prev_time = curr_tick

                log_writer.writerow([
                    curr_time_log,
                    result["action"],
                    confidence,
                    round(fps, 2)
                ])

            # =========================
            # DRAW
            # =========================
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

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
        # UI
        # =========================
        cv2.rectangle(
            image,
            (0, 0),
            (700, 120),
            (245, 117, 16),
            -1
        )

        cv2.putText(
            image,
            f"Action: {action}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
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

        cv2.putText(
            image,
            f"Sequence: {len(sequence)}/{SEQ_LENGTH}",
            (350, 85),
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

        key = cv2.waitKey(1)

        if key & 0xFF == ord('q') or key == 27:
            break

# =========================
# EXIT
# =========================
cap.release()

log_file.close()

cv2.destroyAllWindows()

print("로그 저장 완료")