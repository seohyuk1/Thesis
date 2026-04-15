import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

# 각도 계산 함수
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

file_raw_squat = open('Data Collection/csv/raw/squat.csv', 'a', newline='')
writer_raw_squat = csv.writer(file_raw_squat)

# EVENT
file_event_all = open('Data Collection/csv/event/reps_data.csv', 'a', newline='')
writer_event_all = csv.writer(file_event_all)

file_event_squat = open('Data Collection/csv/event/squat_reps.csv', 'a', newline='')
writer_event_squat = csv.writer(file_event_squat)

# 헤더
headers = ['time','angle','count','stage','mode']

for f, w in [
    (file_raw_all, writer_raw_all),
    (file_raw_squat, writer_raw_squat),
    (file_event_all, writer_event_all),
    (file_event_squat, writer_event_squat)
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

cv2.namedWindow('Squat Data Collector', cv2.WINDOW_NORMAL)

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
            
            hip = [
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            ]
            
            knee = [
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y
            ]
            
            ankle = [
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y
            ]
            
            angle = calculate_angle(hip, knee, ankle)
            
            if angle > 160:
                stage = "up"
            
            if angle < 90 and stage == "up":
                stage = "down"
                counter += 1
            
            h, w, _ = image.shape
            cv2.putText(
                image,
                str(int(angle)),
                tuple(np.multiply(knee, [w, h]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )
                
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
                "squat"
            ]

            # RAW (0.1초)
            if current_time - last_save_time > 0.1:
                writer_raw_all.writerow(row)
                writer_raw_squat.writerow(row)
                last_save_time = current_time

            # EVENT (카운트 증가)
            if counter != prev_counter:
                writer_event_all.writerow(row)
                writer_event_squat.writerow(row)
                prev_counter = counter
        
        # UI
        cv2.rectangle(image, (0, 0), (250, 80), (245, 117, 16), -1)
        
        cv2.putText(image, 'SQUAT', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        cv2.putText(image, 'REPS', (10, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(image, str(counter), (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        
        cv2.putText(image, 'STAGE', (120, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        cv2.putText(image, str(stage), (120, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        
        resized_image = cv2.resize(image, (960, 720))
        cv2.imshow('Squat Data Collector', resized_image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# 종료
file_raw_all.close()
file_raw_squat.close()
file_event_all.close()
file_event_squat.close()
cap.release()
cv2.destroyAllWindows()