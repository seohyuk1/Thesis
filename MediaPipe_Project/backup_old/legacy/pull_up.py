# ❗ DEPRECATED (사용 안함)
# exercise_selection.py 사용 권장

import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

# =========================
# 저장 경로
# =========================
SAVE_DIR = 'data/landmark/raw'
os.makedirs(SAVE_DIR, exist_ok=True)

mode = "pullup"

# =========================
# 자동 파일 이름 생성
# =========================
existing_files = [f for f in os.listdir(SAVE_DIR) if f.startswith(mode)]
file_index = len(existing_files) + 1

file_name = f"{mode}_{file_index:02d}.csv"
file_path = os.path.join(SAVE_DIR, file_name)

print(f"저장 파일: {file_name}")

file = open(file_path, 'w', newline='')  # 'a' → 'w' (새 파일 생성)
writer = csv.writer(file)

# =========================
# 헤더 생성
# =========================
headers = ['time', 'mode']

for i in range(33):
    headers += [f'x{i}', f'y{i}', f'z{i}']

writer.writerow(headers)  # 항상 새 파일이므로 바로 작성

# =========================
# Mediapipe 설정
# =========================
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Pull-up Data Collector', cv2.WINDOW_NORMAL)

start_time = time.time()
last_save_time = 0

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark

            current_time = time.time() - start_time

            # =========================
            # landmark row 생성
            # =========================
            row = [
                round(current_time, 4),
                mode
            ]

            for lm in landmarks:
                row.extend([lm.x, lm.y, lm.z])

            # =========================
            # 저장 (0.1초 간격)
            # =========================
            if current_time - last_save_time > 0.1:
                writer.writerow(row)
                last_save_time = current_time

        except:
            pass
        
        # =========================
        # 시각화
        # =========================
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        resized = cv2.resize(image, (960,720))
        cv2.imshow('Pull-up Data Collector', resized)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 종료
file.close()
cap.release()
cv2.destroyAllWindows()