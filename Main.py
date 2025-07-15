import cv2
import mediapipe as mp
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import pygame
import time
import queue

# Constants
EAR_THRESHOLD = 0.25
CONSEC_FRAMES = 20

# Sound path
sound_path = "alarm.wav"

# Queues and status flags
threadStatusQ = queue.Queue()
ALARM_ON = False
ALERT_THREAD_STARTED = False

# MediaPipe eye indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]   # Approximate for eye aspect ratio
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # Approximate for eye aspect ratio

# EAR calculation
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Sound alert function
def sound_alert(path, threadStatusQ):
    pygame.mixer.init()
    pygame.mixer.music.load(path)
    pygame.mixer.music.play(-1)  # Loop indefinitely

    while True:
        if not threadStatusQ.empty():
            stop_signal = threadStatusQ.get()
            if stop_signal:
                pygame.mixer.music.stop()
                break
        time.sleep(0.1)

# Main function
def main():
    global ALARM_ON, ALERT_THREAD_STARTED

    print("[INFO] Starting Drowsiness Detection... Press 'q' or ESC to quit.")

    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    counter = 0
    blink_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = face_landmarks.landmark

                # Get eye landmarks
                left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
                right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]

                # EAR
                leftEAR = eye_aspect_ratio(left_eye)
                rightEAR = eye_aspect_ratio(right_eye)
                ear = (leftEAR + rightEAR) / 2.0

                # Draw eyes
                for (x, y) in left_eye + right_eye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

                if ear < EAR_THRESHOLD:
                    counter += 1
                    if counter >= CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        if not ALARM_ON:
                            ALARM_ON = True
                            if not ALERT_THREAD_STARTED:
                                threadStatusQ.queue.clear()
                                thread = Thread(target=sound_alert, args=(sound_path, threadStatusQ))
                                thread.daemon = True
                                thread.start()
                                ALERT_THREAD_STARTED = True
                else:
                    if counter >= 2:
                        blink_count += 1
                    counter = 0
                    if ALARM_ON:
                        ALARM_ON = False
                        threadStatusQ.put(True)
                        ALERT_THREAD_STARTED = False

                cv2.putText(frame, f"Blinks: {blink_count}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        else:
            cv2.putText(frame, "Face not detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Drowsiness Detection", frame)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Exiting...")

if __name__ == "__main__":
    main()
