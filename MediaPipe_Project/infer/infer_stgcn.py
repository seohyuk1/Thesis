import cv2
import mediapipe as mp
import numpy as np
from tensorflow import keras
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = os.path.join("models", "saved", "stgcn_model.keras")

SEQ_LENGTH = 30  # train과 동일
NUM_JOINTS = 33
NUM_COORDS = 3

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
print("ST-GCN Model loaded!")

# =========================
# Mediapipe
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

sequence = []

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

        # 이미지 처리
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        action = "Waiting..."

        try:
            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark

                # (33,3) 구조 그대로 유지
                frame_data = []
                for lm in landmarks:
                    frame_data.append([lm.x, lm.y, lm.z])

                sequence.append(frame_data)

                # 시퀀스 유지
                if len(sequence) > SEQ_LENGTH:
                    sequence.pop(0)

                # =========================
                # 예측
                # =========================
                if len(sequence) == SEQ_LENGTH:
                    input_data = np.array(sequence)  # (T, V, C)
                    input_data = np.expand_dims(input_data, axis=0)  # (1, T, V, C)

                    pred = model.predict(input_data, verbose=0)
                    pred_class = np.argmax(pred)

                    action = LABEL_MAP.get(pred_class, "unknown")

        except Exception as e:
            print("에러:", e)

        # =========================
        # UI
        # =========================
        cv2.rectangle(image, (0, 0), (320, 80), (245, 117, 16), -1)

        cv2.putText(image, f'Action: {action}',
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2)

        # 랜드마크
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        cv2.imshow("ST-GCN Inference", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 종료
cap.release()
cv2.destroyAllWindows()