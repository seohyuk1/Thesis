import cv2
import mediapipe as mp

# 1. 사용할 도구들 가져오기
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 2. 웹캠 설정
cap = cv2.VideoCapture(0)

# 3. 손 인식 모델 실행
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    print("웹캠이 켜집니다... 종료하려면 ESC를 누르세요.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("웹캠을 찾을 수 없습니다.")
            break

        # 처리 속도 향상을 위해 좌우 반전 및 RGB 변환
        image = cv2.flip(image, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe 처리
        results = hands.process(image_rgb)

        # 손이 발견되면 화면에 그리기
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # 결과 화면 표시
        cv2.imshow('MediaPipe Hands Success!', image)

        # ESC 누르면 종료
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()