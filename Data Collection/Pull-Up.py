import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

# =========================
# CSV 폴더 구조
# =========================
os.makedirs('Data Collection/csv/raw', exist_ok=True)
os.makedirs('Data Collection/csv/event', exist_ok=True)

# RAW
file_raw_all = open('Data Collection/csv/raw/exercise_data.csv', 'a', newline='')
writer_raw_all = csv.writer(file_raw_all)

file_raw_pull = open('Data Collection/csv/raw/pullup.csv', 'a', newline='')
writer_raw_pull = csv.writer(file_raw_pull)

# EVENT
file_event_all = open('Data Collection/csv/event/reps_data.csv', 'a', newline='')
writer_event_all = csv.writer(file_event_all)

file_event_pull = open('Data Collection/csv/event/pullup_reps.csv', 'a', newline='')
writer_event_pull = csv.writer(file_event_pull)

# 헤더
headers = ['time','angle','count','stage','mode']

for f, w in [
    (file_raw_all, writer_raw_all),
    (file_raw_pull, writer_raw_pull),
    (file_event_all, writer_event_all),
    (file_event_pull, writer_event_pull)
]:
    if f.tell() == 0:
        w.writerow(headers)

# Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 카메라
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Pull-up Data Collector', cv2.WINDOW_NORMAL)

counter = 0
prev_counter = 0   # event용
stage = None

start_time = time.time()
last_save_time = 0  # raw 간격

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
        
        angle = None
        
        try:
            landmarks = results.pose_landmarks.landmark
            h, w, _ = image.shape
            
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]

            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]

            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

            angle = calculate_angle(shoulder, elbow, wrist)

            # 카운트 로직
            if angle > 150:
                stage = "down"
            
            if angle < 70 and stage == "down":
                stage = "up"
                counter += 1

            # 각도 표시
            cv2.putText(image, str(int(angle)),
                        tuple(np.multiply(elbow, [w, h]).astype(int)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)

        except:
            pass
        
        # =========================
        # CSV 저장
        # =========================
        if angle is not None:
            current_time = time.time() - start_time

            row = [
                round(current_time, 4),
                round(angle, 2),
                counter,
                stage,
                "pullup"
            ]

            # RAW (0.1초)
            if current_time - last_save_time > 0.1:
                writer_raw_all.writerow(row)
                writer_raw_pull.writerow(row)
                last_save_time = current_time

            # EVENT (카운트 증가)
            if counter != prev_counter:
                writer_event_all.writerow(row)
                writer_event_pull.writerow(row)
                prev_counter = counter
        
        # UI
        cv2.rectangle(image, (0,0), (250,80), (245,117,16), -1)
        
        cv2.putText(image, 'PULL-UP', (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        
        cv2.putText(image, 'REPS', (10,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        cv2.putText(image, str(counter), (10,75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        
        cv2.putText(image, 'STAGE', (120,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        cv2.putText(image, str(stage), (120,75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        resized = cv2.resize(image, (960,720))
        cv2.imshow('Pull-up Data Collector', resized)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 종료
file_raw_all.close()
file_raw_pull.close()
file_event_all.close()
file_event_pull.close()
cap.release()
cv2.destroyAllWindows()