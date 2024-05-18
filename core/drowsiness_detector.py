import cv2
from utils.config import project_config as pj
from keras.utils import img_to_array
import time
import numpy as np
from playsound import playsound
from threading import Thread


def start_alarm(sound):
    """Play the alarm sound"""
    playsound(pj.ALARM_SOUND)


class DrowsinessDetector:
    def __init__(self, detect_model, clf_model) -> None:
        self.detect_model = detect_model
        self.clf_model = clf_model
        self.eye_status1 = None
        self.eye_status2 = None
        self.start_time = 0
        self.end_time = 0
        self.count_start = 0
        self.time_close_eyes = 0
        self.load_model = 0
        self.alarm_on = 0

        self.face_cascade = None
        self.left_eye_cascade = None
        self.right_eye_cascade = None
        self.load_detect_model(detect_model)

    def load_detect_model(self, detect_model):
        if detect_model == "cascade":
            self.face_cascade = cv2.CascadeClassifier(pj.FACE_CASCADE_PATH)
            self.left_eye_cascade = cv2.CascadeClassifier(pj.LEFT_EYE_CASCADE_PATH)
            self.right_eye_cascade = cv2.CascadeClassifier(pj.RIGHT_EYE_CASCADE_PATH)
        self.load_model = True

    def process_eye_frame(self, eye_frame):
        processed_eye_frame = cv2.resize(eye_frame, (145, 145))
        processed_eye_frame = processed_eye_frame.astype("float") / 255.0
        processed_eye_frame = img_to_array(processed_eye_frame)
        processed_eye_frame = np.expand_dims(processed_eye_frame, axis=0)
        return processed_eye_frame

    def get_eye_status(self, frame):
        if self.load_model:
            framegray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(framegray, 1.3, 5)

            for x, y, w, h in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

                roi_gray = framegray[y : y + h, x : x + w]
                roi_color = frame[y : y + h, x : x + w]
                left_eye = self.left_eye_cascade.detectMultiScale(roi_gray)
                right_eye = self.right_eye_cascade.detectMultiScale(roi_gray)
                self.eye_status1 = None
                self.eye_status2 = None
                for x1, y1, w1, h1 in left_eye:
                    cv2.rectangle(
                        roi_color, (x1, y1), (x1 + w1, y1 + h1), (0, 255, 0), 2
                    )
                    eye1 = roi_color[y1 : y1 + h1, x1 : x1 + w1]
                    pred1 = self.clf_model.predict(self.process_eye_frame(eye1))
                    self.eye_status1 = np.argmax(pred1)
                    break

                for x2, y2, w2, h2 in right_eye:
                    cv2.rectangle(
                        roi_color, (x2, y2), (x2 + w2, y2 + h2), (0, 255, 0), 2
                    )
                    eye2 = roi_color[y2 : y2 + h2, x2 : x2 + w2]
                    pred2 = self.clf_model.predict(self.process_eye_frame(eye2))
                    self.eye_status2 = np.argmax(pred2)
                    break
        return frame

    def detect_drosiness(self, frame):
        self.get_eye_status(frame)
        print(self.eye_status1)
        print(self.eye_status2)

        if self.eye_status1 == 1 and self.eye_status2 == 1:
            if not self.count_start:
                self.start_time = time.time()
                self.count_start = True
            if self.count_start:
                self.end_time = time.time()
                self.time_close_eyes = self.end_time - self.start_time

            if self.time_close_eyes >= pj.TIME_THRESHOLD:
                if not self.alarm_on:
                    self.alarm_on = True
                    # play the alarm sound in a new thread
                    t = Thread(target=start_alarm, args=(pj.ALARM_SOUND))
                    t.daemon = True
                    t.start()
                cv2.putText(
                    frame,
                    "Drowsiness Alert!!!",
                    (100, 700),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    1,
                )
            else:
                cv2.putText(
                    frame,
                    "Eyes Closed, time: " + str(round(self.time_close_eyes, 2)),
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 255),
                    1,
                )

        if self.eye_status1 == 0 or self.eye_status2 == 0:
            cv2.putText(
                frame,
                "Eyes Open",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 0),
                1,
            )
            self.time_close_eyes = 0
            self.count_start = False
        return frame