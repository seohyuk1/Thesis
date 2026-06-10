import cv2
import mediapipe as mp
import csv
import os
import time

# =========================
# SAVE PATH
# =========================
SAVE_DIR = "data/raw/landmark"
os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# INITIAL MODE
# =========================
mode = "side_lunge"

current_file = None
writer = None

# =========================
# RECORDING
# =========================
recording = False

# =========================
# CREATE FILE
# =========================
def create_new_file(mode):

    global current_file
    global writer

    if current_file is not None:
        current_file.close()

    timestamp = time.strftime("%Y%m%d_%H%M%S")

    file_name = f"{mode}_{timestamp}.csv"

    file_path = os.path.join(
        SAVE_DIR,
        file_name
    )

    print(f"\n새 파일 생성: {file_name}")

    current_file = open(
        file_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(current_file)

    header = [
        "time",
        "mode"
    ]

    for i in range(33):

        header += [
            f"x{i}",
            f"y{i}",
            f"z{i}"
        ]

    writer.writerow(header)

# =========================
# MEDIAPIPE
# =========================
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# =========================
# CAMERA
# =========================
cap = cv2.VideoCapture(0)

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)

WINDOW_NAME = "Exercise Collector"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    WINDOW_NAME,
    1280,
    720
)

# =========================
# TIMER
# =========================
start_time = time.time()
last_save_time = 0

# =========================
# FPS
# =========================
prev_time = time.time()

# =========================
# LOOP
# =========================
with mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as pose:

    while cap.isOpened():

        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(
            cv2.flip(frame, 1),
            cv2.COLOR_BGR2RGB
        )

        results = pose.process(image)

        image = cv2.cvtColor(
            image,
            cv2.COLOR_RGB2BGR
        )

        try:

            if results.pose_landmarks:

                landmarks = results.pose_landmarks.landmark

                current_time = (
                    time.time() - start_time
                )

                row = [
                    round(current_time, 4),
                    mode
                ]

                for lm in landmarks:

                    row.extend([
                        lm.x,
                        lm.y,
                        lm.z
                    ])

                # =====================
                # SAVE CSV
                # =====================
                if recording and (
                    current_time - last_save_time > 0.1
                ):

                    writer.writerow(row)

                    last_save_time = current_time

                mp_drawing.draw_landmarks(
                    image,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

        except Exception as e:

            print("에러:", e)

        # =====================
        # FPS
        # =====================
        curr_time = time.time()

        fps = (
            1 / (curr_time - prev_time)
            if curr_time != prev_time
            else 0
        )

        prev_time = curr_time

        # =====================
        # UI
        # =====================
        cv2.rectangle(
            image,
            (0, 0),
            (900, 150),
            (245, 117, 16),
            -1
        )

        cv2.putText(
            image,
            f"MODE: {mode}",
            (10, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            2
        )

        cv2.putText(
            image,
            f"RECORDING: {recording}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        cv2.putText(
            image,
            f"FPS: {int(fps)}",
            (10, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

        cv2.putText(
            image,
            "1:Lunge 2:Burpee 3:Plank 4:Pushup 5:Crunch",
            (10, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1
        )

        cv2.imshow(
            WINDOW_NAME,
            image
        )

        key = cv2.waitKey(10) & 0xFF

        # =====================
        # MODE CHANGE
        # =====================
        if not recording:

            if key == ord('1'):
                mode = "side_lunge"

            elif key == ord('2'):
                mode = "burpee"

            elif key == ord('3'):
                mode = "plank"

            elif key == ord('4'):
                mode = "pushup"

            elif key == ord('5'):
                mode = "crunch"

        # =====================
        # RECORD START/STOP
        # =====================
        if key == 32:

            if not recording:

                create_new_file(mode)

                recording = True

                print(
                    f"\n[{mode}] 녹화 시작"
                )

            else:

                recording = False

                print(
                    f"\n[{mode}] 녹화 종료"
                )

                if current_file:
                    current_file.close()
                    current_file = None

        # =====================
        # EXIT
        # =====================
        elif key == ord('q') or key == 27:
            break

# =========================
# EXIT
# =========================
if current_file:
    current_file.close()

cap.release()

cv2.destroyAllWindows()

print("\n데이터 수집 종료")