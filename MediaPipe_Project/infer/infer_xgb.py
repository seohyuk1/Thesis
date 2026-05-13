import cv2
import mediapipe as mp
import numpy as np
import joblib
import os

# =========================
# CONFIG
# =========================
MODEL_PATH = os.path.join("models", "saved", "xgb_model.pkl")

WINDOW_SIZE = 15  # train과 동일

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
print("XGB Model loaded!")

# =========================
# FEATURE FUNCTION (train과 동일)
# =========================
def extract_features(sequence):
    sequence = np.array(sequence)

    features = []

    features.append(np.mean(sequence))
    features.append(np.std(sequence))
    features.append(np.max(sequence))
    features.append(np.min(sequence))

    if len(sequence) > 1:
        diff = np.diff(sequence, axis=0)
        features.append(np.mean(diff))
        features.append(np.std(diff))
    else:
        features.append(0)
        features.append(0)

    for i in range(sequence.shape[1]):
        features.append(np.mean(sequence[:, i]))
        features.append(np.std(sequence[:, i]))

    return np.array(features)


# =========================
# Mediapipe 설정
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

# 시퀀스 버퍼
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

                # (33,3) → (99,)
                row = []
                for lm in landmarks:
                    row.extend([lm.x, lm.y, lm.z])

                sequence.append(row)

                # window 유지
                if len(sequence) > WINDOW_SIZE:
                    sequence.pop(0)

                # 예측
                if len(sequence) == WINDOW_SIZE:
                    features = extract_features(sequence).reshape(1, -1)

                    pred = model.predict(features)[0]
                    action = LABEL_MAP.get(pred, "unknown")

        except Exception as e:
            print("에러:", e)

        # =========================
        # UI 표시
        # =========================
        cv2.rectangle(image, (0, 0), (300, 80), (245, 117, 16), -1)

        cv2.putText(image, f'Action: {action}',
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2)

        # 랜드마크 그리기
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        cv2.imshow("XGB Inference", image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 종료
cap.release()
cv2.destroyAllWindows()