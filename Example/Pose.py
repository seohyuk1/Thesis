import cv2
import mediapipe as mp

# 1. 사용할 도구들 정의
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 2. 웹캠 설정
cap = cv2.VideoCapture(0)

# 3. Pose 인스턴스 생성
# confidence 값들을 조절해 인식 민감도를 설정할 수 있습니다.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    
    print("전신 인식 시작... 종료하려면 'q'를 누르세요.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 4. 이미지 전처리
        # 좌우 반전(셀카 모드) 및 RGB 변환
        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # 5. 자세 검출 수행
        results = pose.process(image)
    
        # 6. 화면 표시를 위해 다시 BGR로 변환
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # 7. 랜드마크 데이터 추출 및 시각화
        if results.pose_landmarks:
            # 관절 포인트와 연결선 그리기
            mp_drawing.draw_landmarks(
                image, 
                results.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), # 점 색상
                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)  # 선 색상
            )
            
            # (선택) 특정 랜드마크 좌표 확인용 (예: 코의 위치)
            # landmarks = results.pose_landmarks.landmark
            # print(landmarks[mp_pose.PoseLandmark.NOSE])

        # 8. 결과 창 띄우기
        cv2.imshow('Mediapipe Pose Feed', image)

        # 'q' 키를 누르면 종료
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()