# %%
import keras
import cv2
import time
from utils.config import project_config as pj
from core import drowsiness_detector


cap = cv2.VideoCapture(0)
model = keras.models.load_model(pj.MODEL_PATH1)
alarm_on = False
show_fps = 1
detector = drowsiness_detector.DrowsinessDetector("yolo", model)

if show_fps:
    num_frames = 0

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break
    start_time = time.time()

    frame = detector.detect_drosiness(frame)

    # Caculate FPS
    if show_fps:
        num_frames += 1
        elapsed_time = time.time() - start_time
        fps = num_frames / elapsed_time
        num_frames = 0
        cv2.putText(
            frame,
            f"FPS: {round(fps, 2)}",
            (510, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imshow("Drowsiness Detector", frame)

    key = cv2.waitKey(1) & 0xFF
    # Check if user presses "q" key or ESC button, exit while loop
    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()
