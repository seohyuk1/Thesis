import cv2
import mediapipe as mp
import numpy as np

# 1. 각도 계산 함수
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

# 🔥 웹캠 설정 + 해상도 지정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 🔥 창 크기 조절 가능하게 설정
cv2.namedWindow('Squat Counter', cv2.WINDOW_NORMAL)

# 카운터 변수
counter = 0
stage = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 좌우 반전 + 색 변환
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        try:
            landmarks = results.pose_landmarks.landmark
            
            # 스쿼트 관절
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
            
            # 각도 계산
            angle = calculate_angle(hip, knee, ankle)
            
            # 각도 표시 (해상도 기준 자동 적용)
            h, w, _ = image.shape
            cv2.putText(
                image,
                str(int(angle)),
                tuple(np.multiply(knee, [w, h]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )
            
            # 스쿼트 카운트
            if angle > 160:
                stage = "up"
            
            if angle < 90 and stage == "up":
                stage = "down"
                counter += 1
                
        except:
            pass
        
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
        
        # 랜드마크
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )
        
        # 🔥 화면 크기 자동 조절 (핵심)
        resized_image = cv2.resize(image, (960, 720))
        
        cv2.imshow('Squat Counter', resized_image)
        
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()