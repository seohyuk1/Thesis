import cv2
import mediapipe as mp
import numpy as np
import csv
import time
import os

# 운동 선택
mode = "squat"

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
# CSV 설정
# =========================
os.makedirs('Data Collection/csv', exist_ok=True)

# raw 데이터
file_all = open('Data Collection/csv/exercise_data.csv', 'a', newline='')
writer_all = csv.writer(file_all)

# 운동별
file_push = open('Data Collection/csv/pushup.csv', 'a', newline='')
writer_push = csv.writer(file_push)

file_pull = open('Data Collection/csv/pullup.csv', 'a', newline='')
writer_pull = csv.writer(file_pull)

file_squat = open('Data Collection/csv/squat.csv', 'a', newline='')
writer_squat = csv.writer(file_squat)

# event 데이터
file_event = open('Data Collection/csv/event_data.csv', 'a', newline='')
writer_event = csv.writer(file_event)

# 헤더
if file_all.tell() == 0:
    writer_all.writerow(['time','angle','count','stage','mode'])

if file_push.tell() == 0:
    writer_push.writerow(['time','angle','count','stage','mode'])

if file_pull.tell() == 0:
    writer_pull.writerow(['time','angle','count','stage','mode'])

if file_squat.tell() == 0:
    writer_squat.writerow(['time','angle','count','stage','mode'])

if file_event.tell() == 0:
    writer_event.writerow(['rep','mode','duration','min_angle','max_angle'])

# Mediapipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 카메라
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow('Exercise Data Collector', cv2.WINDOW_NORMAL)

counter = 0
stage = None

# event용 변수
rep_start_time = None
angles = []

start_time = time.time()

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

            # =========================
            # 운동별 처리
            # =========================
            if mode == "squat":
                hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x,
                       landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
                knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x,
                        landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
                ankle = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y]

                angle = calculate_angle(hip, knee, ankle)

                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == "up":
                    stage = "down"
                    counter += 1
                    event_trigger = True
                else:
                    event_trigger = False

                joint_point = knee

            elif mode == "pushup":
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 160:
                    stage = "up"
                if angle < 90 and stage == "up":
                    stage = "down"
                    counter += 1
                    event_trigger = True
                else:
                    event_trigger = False

                joint_point = elbow

            elif mode == "pullup":
                shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                            landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                angle = calculate_angle(shoulder, elbow, wrist)

                if angle > 150:
                    stage = "down"
                if angle < 70 and stage == "down":
                    stage = "up"
                    counter += 1
                    event_trigger = True
                else:
                    event_trigger = False

                joint_point = elbow

            # 각도 표시
            if angle is not None:
                angles.append(angle)

                cv2.putText(
                    image,
                    str(int(angle)),
                    tuple(np.multiply(joint_point, [w, h]).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255,255,255),
                    2
                )

        except:
            event_trigger = False
        
        # =========================
        # RAW CSV 저장
        # =========================
        if angle is not None:
            current_time = time.time() - start_time

            row = [
                round(current_time, 4),
                round(angle, 2),
                counter,
                stage,
                mode
            ]

            writer_all.writerow(row)

            if mode == "pushup":
                writer_push.writerow(row)
            elif mode == "pullup":
                writer_pull.writerow(row)
            elif mode == "squat":
                writer_squat.writerow(row)

        # =========================
        # EVENT 저장 (핵심)
        # =========================
        if event_trigger:
            now = time.time()

            if rep_start_time is not None and len(angles) > 0:
                duration = now - rep_start_time
                min_angle = min(angles)
                max_angle = max(angles)

                writer_event.writerow([
                    counter,
                    mode,
                    round(duration, 3),
                    round(min_angle, 2),
                    round(max_angle, 2)
                ])

            rep_start_time = now
            angles = []

        # UI
        cv2.rectangle(image, (0,0), (300,100), (245,117,16), -1)
        
        cv2.putText(image, mode.upper(), (10,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        cv2.putText(image, str(counter), (150,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        
        cv2.putText(image, str(stage), (230,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        resized_image = cv2.resize(image, (960, 720))
        cv2.imshow('Exercise Data Collector', resized_image)
        
        key = cv2.waitKey(10) & 0xFF
        
        if key == ord('1'):
            mode = "squat"
            counter = 0
        
        elif key == ord('2'):
            mode = "pushup"
            counter = 0
        
        elif key == ord('3'):
            mode = "pullup"
            counter = 0
        
        elif key == ord('q'):
            break

# 종료
file_all.close()
file_push.close()
file_pull.close()
file_squat.close()
file_event.close()
cap.release()
cv2.destroyAllWindows()