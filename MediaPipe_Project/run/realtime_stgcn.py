import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from collections import deque
import os
import csv
import time

# =========================
# PATH
# =========================
BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "saved",
    "stgcn_model.keras"
)

# =========================
# LOAD MODEL
# =========================
if not os.path.exists(MODEL_PATH):

    print("모델 파일 없음:", MODEL_PATH)
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

print("ST-GCN 모델 로드 완료")

# =========================
# LABEL
# =========================
LABEL_MAP = {
    0: "squat",
    1: "pushup",
    2: "pullup"
}

# =========================
# LOG
# =========================
LOG_DIR = os.path.join(
    BASE_DIR,
    "logs",
    "stgcn"
)

os.makedirs(LOG_DIR, exist_ok=True)

LOG_PATH = os.path.join(
    LOG_DIR,
    "realtime_log.csv"
)

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

WINDOW_NAME = "Realtime Action Recognition (ST-GCN)"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1280,
    720
)

# =========================
# ST-GCN CONFIG
# =========================
SEQ_LENGTH = 30

sequence = deque(
    maxlen=SEQ_LENGTH
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

        # =========================
        # IMAGE
        # =========================
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

            landmarks = results.pose_landmarks.landmark

            frame_data = []

            for lm in landmarks:

                frame_data.append([
                    lm.x,
                    lm.y,
                    lm.z
                ])

            frame_data = np.array(
                frame_data,
                dtype=np.float32
            )

            # (33,3)
            sequence.append(frame_data)

            # draw
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS
            )

        # =========================
        # PREDICT
        # =========================
        if len(sequence) == SEQ_LENGTH:

            input_data = np.array(
                sequence,
                dtype=np.float32
            )

            # (1,30,33,3)
            input_data = np.expand_dims(
                input_data,
                axis=0
            )

            pred = model.predict(
                input_data,
                verbose=0
            )[0]

            pred_class = np.argmax(pred)

            confidence = float(
                np.max(pred)
            )

            action_name = LABEL_MAP.get(
                pred_class,
                "unknown"
            )

            action = (
                f"{action_name} "
                f"({confidence:.2f})"
            )

            # =========================
            # SAVE LOG
            # =========================
            log_writer.writerow([
                time.time(),
                action_name,
                round(confidence, 4),
                0
            ])

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
            (550, 110),
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

        cv2.putText(
            image,
            f"Sequence: {len(sequence)}/{SEQ_LENGTH}",
            (300, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
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

        # =========================
        # EXIT
        # =========================
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