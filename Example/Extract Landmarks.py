import cv2
import mediapipe as mp
import numpy as np

# 1. 각도를 계산하는 함수 정의 (세 점 사이의 각도 구하기)
def calculate_angle(a, b, c):
    a = np.array(a) # 첫 번째 점 (예: 어깨)
    b = np.array(b) # 두 번째 점 (예: 팔꿈치)
    c = np.array(c) # 세 번째 점 (예: 손목)
    
    # 아크탄젠트를 이용해 라디안 값을 구하고 도(degree)로 변환
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    # 각도가 180도를 넘지 않도록 조정
    if angle > 180.0:
        angle = 360-angle
        
    return angle 

# 2. MediaPipe 도구 로드
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 3. 웹캠 연결
cap = cv2.VideoCapture(0)

# 4. Pose 인스턴스 실행
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 이미지 전처리 (RGB 변환 및 좌우 반전)
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # 감지 수행
        results = pose.process(image)
    
        # 다시 그리기를 위해 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 5. 랜드마크 추출 및 각도 계산
        try:
            landmarks = results.pose_landmarks.landmark
            
            # 왼쪽 어깨, 팔꿈치, 손목의 x, y 좌표 추출
            shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x, 
                        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
            elbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
            wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x, 
                     landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
            
            # 각도 계산 함수 호출
            angle = calculate_angle(shoulder, elbow, wrist)
            
            # 화면에 각도 숫자 표시
            # np.multiply(elbow, [640, 480])는 0~1 사이의 좌표를 실제 픽셀 위치로 바꿔줍니다.
            cv2.putText(image, str(int(angle)), 
                           tuple(np.multiply(elbow, [640, 480]).astype(int)), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
                               
        except Exception as e:
            # 관절이 화면 밖으로 나가서 인식이 안 될 때 에러 방지
            pass
        
        # 6. 관절 포인트 및 연결선 그리기
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )               
        
        # 결과 화면 출력
        cv2.imshow('Mediapipe Pose Angle Feed', image)

        # 'q'를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()