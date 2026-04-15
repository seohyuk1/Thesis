import cv2
import mediapipe as mp
import numpy as np

# 🔥 운동 선택 ("squat", "pushup", "pullup")
mode = "squat"

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

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 🔥 웹캠 설정 + 해상도
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 🔥 창 크기 조절 가능
cv2.namedWindow('Exercise Counter', cv2.WINDOW_NORMAL)

# 카운트 변수
counter = 0
stage = None

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
            
            h, w, _ = image.shape

            # =========================
            # 🔥 SQUAT
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

                joint_point = knee

            # =========================
            # 🔥 PUSH-UP
            # =========================
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

                joint_point = elbow

            # =========================
            # 🔥 PULL-UP
            # =========================
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

                joint_point = elbow

            # 🔥 각도 표시
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
            pass
        
        # =========================
        # 🔥 UI
        # =========================
        cv2.rectangle(image, (0,0), (300,100), (245,117,16), -1)
        
        cv2.putText(image, 'MODE', (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        cv2.putText(image, mode.upper(), (10,45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        cv2.putText(image, 'REPS', (150,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        cv2.putText(image, str(counter), (150,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,255,255), 2)
        
        cv2.putText(image, 'STAGE', (230,20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        cv2.putText(image, str(stage), (230,70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # 랜드마크
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        
        # 🔥 화면 resize
        resized_image = cv2.resize(image, (960, 720))
        cv2.imshow('Exercise Counter', resized_image)
        
        # 🔥 키 입력으로 운동 변경
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

cap.release()
cv2.destroyAllWindows()