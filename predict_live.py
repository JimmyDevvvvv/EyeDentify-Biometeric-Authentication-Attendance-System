# predict_live.py

import os
import cv2
import torch
import numpy as np
from model.iris_net import IrisNet
from utils.preprocess import preprocess_image

# =========================
# Settings
# =========================
MODEL_PATH = "model/iris_model.pth"
DATASET_PATH = "dataset"
IMG_SIZE = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# Load Model
# =========================
users = sorted(os.listdir(DATASET_PATH))
label_map = {i: user for i, user in enumerate(users)}
num_classes = len(label_map)

model = IrisNet(num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

print("[INFO] Model loaded. Starting webcam...")

# =========================
# Start Webcam
# =========================
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)  # Mirror view

    # Eye ROI box
    h, w = frame.shape[:2]
    box_size = 200
    x1 = w // 2 - box_size // 2
    y1 = h // 2 - box_size // 2
    x2 = x1 + box_size
    y2 = y1 + box_size
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Extract & preprocess eye ROI
    eye = frame[y1:y2, x1:x2]
    gray_eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray_eye, (IMG_SIZE, IMG_SIZE)) / 255.0
    tensor = torch.tensor(resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1, 1, 100, 100)

    # Predict
    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)
        name = label_map[predicted.item()]
        confidence = confidence.item()

    # Show result
    if confidence > 0.8:
        msg = f"✅ Welcome, {name} ({confidence:.2f})"
        color = (0, 255, 0)
    else:
        msg = f"❌ Unrecognized ({confidence:.2f})"
        color = (0, 0, 255)

    cv2.putText(frame, msg, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.imshow("EyeDentify Live Scanner", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
