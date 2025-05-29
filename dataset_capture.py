# dataset_capture.py

import cv2
import os

DATASET_PATH = "dataset"
NUM_IMAGES = 30  # Number of images per user

def capture_images(username):
    user_path = os.path.join(DATASET_PATH, username)
    os.makedirs(user_path, exist_ok=True)

    cap = cv2.VideoCapture(0)
    count = 0

    print(f"[INFO] Starting capture for {username}. Press 'q' to quit early.")
    
    while count < NUM_IMAGES:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # Mirror effect for your beautiful face

        # Draw a square in the center as a guide
        h, w = frame.shape[:2]
        box_size = 200
        x1 = w // 2 - box_size // 2
        y1 = h // 2 - box_size // 2
        x2 = x1 + box_size
        y2 = y1 + box_size
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop and save eye region inside the box
        eye_img = frame[y1:y2, x1:x2]
        img_path = os.path.join(user_path, f"{count+1}.jpg")
        cv2.imwrite(img_path, eye_img)

        cv2.putText(frame, f"Captured: {count+1}/{NUM_IMAGES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Capturing Iris", frame)
        count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"[INFO] Capture complete for {username}. Saved in '{user_path}'")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    username = input("Enter username: ").strip().lower()
    capture_images(username)
