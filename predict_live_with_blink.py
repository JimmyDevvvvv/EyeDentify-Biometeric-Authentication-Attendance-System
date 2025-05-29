import cv2
import torch
import numpy as np
import mediapipe as mp
import os
import csv
from datetime import datetime
from model.iris_net import IrisNet
from utils.preprocess import preprocess_image

# ========== Config ========== #
EAR_THRESHOLD = 0.21
CONSEC_FRAMES = 3
IMG_SIZE = 100
MODEL_PATH = "model/iris_model.pth"
DATASET_PATH = "dataset"
LOG_FILE = "attendance_log.csv"

# ========== Load Model ========== #
users = sorted(os.listdir(DATASET_PATH))
label_map = {i: user for i, user in enumerate(users)}
num_classes = len(label_map)

model = IrisNet(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ========== Blink Detection Setup ========== #
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [263, 387, 385, 362, 380, 373]

def eye_aspect_ratio(landmarks, eye_indices, w, h):
    p = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_indices]
    A = np.linalg.norm(np.array(p[1]) - np.array(p[5]))
    B = np.linalg.norm(np.array(p[2]) - np.array(p[4]))
    C = np.linalg.norm(np.array(p[0]) - np.array(p[3]))
    return (A + B) / (2.0 * C)

def draw_clean_box(frame, x1, y1, x2, y2, color):
    thickness = 2
    cs = 20
    cv2.line(frame, (x1, y1), (x1 + cs, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + cs), color, thickness)
    cv2.line(frame, (x2, y1), (x2 - cs, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + cs), color, thickness)
    cv2.line(frame, (x1, y2), (x1 + cs, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - cs), color, thickness)
    cv2.line(frame, (x2, y2), (x2 - cs, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - cs), color, thickness)

def draw_status_info(frame, h, w, msg, color, confidence=0, blink_status=""):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-70), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, msg, (20, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    if confidence > 0:
        cv2.putText(frame, f"Confidence: {confidence:.1%}", (20, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    if blink_status:
        cv2.putText(frame, blink_status, (w-200, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)

def draw_header(frame, w):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, "EyeDentify", (20, 35), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "Biometric Attendance", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

def log_attendance(name):
    today = datetime.now().strftime("%Y-%m-%d")
    if not os.path.exists(LOG_FILE):
        with open(LOG_FILE, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Name", "Timestamp", "Date"])

    with open(LOG_FILE, mode="r") as f:
        lines = f.readlines()
        if any(name in line and today in line for line in lines):
            return

    with open(LOG_FILE, mode="a", newline="") as f:
        writer = csv.writer(f)
        timestamp = datetime.now().strftime("%H:%M:%S")
        writer.writerow([name, timestamp, today])

# ========== Start Camera ========== #
cap = cv2.VideoCapture(0)
blink_counter = 0
blink_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_frame)

    draw_header(frame, w)

    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0].landmark
        left_ear = eye_aspect_ratio(face, LEFT_EYE_IDX, w, h)
        right_ear = eye_aspect_ratio(face, RIGHT_EYE_IDX, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            blink_counter += 1
        else:
            if blink_counter >= CONSEC_FRAMES:
                blink_detected = True
            blink_counter = 0

        x1 = w // 2 - 100
        y1 = h // 2 - 100
        x2 = x1 + 200
        y2 = y1 + 200
        eye = frame[y1:y2, x1:x2]
        gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray_eye, (IMG_SIZE, IMG_SIZE)) / 255.0
        tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        with torch.no_grad():
            outputs = model(tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
            name = label_map[predicted.item()]
            confidence = confidence.item()

        if confidence > 0.8 and blink_detected:
            msg = f"Authenticated: {name}"
            color = (0, 255, 0)
            blink_status = "Blink: Verified"
            log_attendance(name)
        elif confidence > 0.8 and not blink_detected:
            msg = f"Please blink to verify"
            color = (0, 255, 255)
            blink_status = "Blink: Required"
        else:
            msg = f"Not recognized"
            color = (0, 0, 255)
            blink_status = "Blink: Pending"

        draw_clean_box(frame, x1, y1, x2, y2, color)
        draw_status_info(frame, h, w, msg, color, confidence, blink_status)

    else:
        blink_detected = False
        msg = "No face detected"
        color = (255, 0, 0)
        x1 = w // 2 - 100
        y1 = h // 2 - 100
        x2 = x1 + 200
        y2 = y1 + 200
        draw_clean_box(frame, x1, y1, x2, y2, color)
        draw_status_info(frame, h, w, msg, color)

    cv2.imshow("EyeDentify - Attendance Mode", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
