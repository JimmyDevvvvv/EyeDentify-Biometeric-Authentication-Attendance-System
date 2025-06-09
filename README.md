
# 👁️ EyeDentify – Blink-Secured Iris Authentication & Attendance System

A lightweight, privacy-aware iris recognition system that uses **blinking** as a liveness check. Built with **PyTorch**, **MediaPipe**, and **OpenCV**, EyeDentify prevents spoofing (e.g., photos) and logs attendance **only when a real, live person is detected**.

---

## 📦 Features

- 🔐 Iris-based biometric authentication  
- 👀 Blink detection to prevent spoofing  
- 🧠 CNN-based iris classification (PyTorch)  
- 📸 Real-time webcam capture  
- ✅ CSV-based attendance logging with de-duplication  
- 🧊 Clean, minimalist GUI overlay

---

## ⚙️ Setup

### 🛠️ Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/EyeDentify-Biometeric-Authentication-Attendance-System.git
cd EyeDentify-Biometeric-Authentication-Attendance-System
```

---

### 🧰 Step 1.5 (Required for First-Time Setup): Create a Virtual Environment & Install Dependencies

We recommend using a virtual environment to keep dependencies clean and isolated.

#### 🪟 On Windows:

```bash
python -m venv venu
.\venu\Scripts\activate
```

#### 🐧 On macOS/Linux:

```bash
python3 -m venv venu
source venu/bin/activate
```

---

### 📦 Install Dependencies Manually

```bash
pip install opencv-python mediapipe torch torchvision numpy
```

> ✅ You only need to run this once after creating the environment.  
> ⚠️ Make sure the virtual environment is activated **every time you run the project** (`(venu)` should appear in your terminal).

---

## 🚀 Usage

### 👤 Step 1: Enroll a New User

```bash
python dataset_capture.py
```

You'll be prompted:

```
Enter username:
```

➡️ Enter a unique name like `ahmed`, `sarah`, or `jimmy`.  
Then:

- Sit in front of the webcam  
- Center your face  
- System will capture **30 eye samples**  
- Folder `dataset/{username}` will be created  

Repeat this step for each person you want to enroll.

---

### 🧠 Step 2: Train the Model

```bash
python train_model.py
```

This will:

- Load all folders under `dataset/`  
- Train a CNN model to recognize each user  
- Save the trained model as:  
  ```
  model/iris_model.pth
  ```
- Automatically generate the class map:  
  ```python
  {0: 'jimmy', 1: 'ahmed', 2: 'sarah'}
  ```

---

### 👁️ Step 3: Run the Blink-Secured Scanner

```bash
python predict_live_with_blink.py
```

Then:

- The webcam launches  
- User must blink to prove liveness  
- If blink and identity match:

```
✅ Authenticated: sarah
```

- If no blink:

```
⚠️ Blink Required
```

- If not in the dataset:

```
❌ Not recognized
```

---

### 📝 Step 4: Attendance Logging

Once a user is successfully authenticated **with a blink**, a line is logged in `attendance_log.csv`:

```
sarah, 2025-05-29 09:14 AM
```

- ✅ Only one log entry per person per day  
- 🧼 De-duplication is handled automatically

---

## 📂 Folder Structure

```
.
├── dataset/                      # Enrolled user folders
│   ├── jimmy/
│   ├── sarah/
│   └── ...
├── model/
│   └── iris_model.pth           # Trained PyTorch model
├── utils/
│   └── preprocess.py            # Image normalization pipeline
├── attendance_log.csv           # Generated log file
├── dataset_capture.py           # Enroll new users
├── train_model.py               # Train model on enrolled users
├── predict_live_with_blink.py  # Live auth with blink detection
├── README.md
```

---

## 🧠 How It Works

| Component       | Function                                  |
|----------------|--------------------------------------------|
| MediaPipe       | Facial & eye landmark detection            |
| OpenCV          | Webcam access + image processing           |
| PyTorch         | CNN for iris image classification          |
| EAR (Eye Ratio) | Used to detect blinking (anti-spoofing)    |
| CSV Logger      | Timestamp-based attendance tracker         |

---

## 🌱 Future Work Ideas

- 🔐 Add **Homomorphic Encryption** to protect iris data  
- 🕵️ Add **Zero-Knowledge Proofs** for anonymous verification  
- 📊 Add a browser-based **dashboard** to view logs  
- 🔄 Add **challenge-response** blinking (e.g., blink twice)  
- 🤖 Use **transfer learning** with more complex CNN architectures  

---

## 👨‍💻 Author

**Built with love by Mohamed Gamal (Jimmy)**  
🛡️ Cybersecurity Invoker

---

## 📜 License

This project is licensed under the **MIT License**.  
Feel free to use, modify, or enhance it — just give credit 🙌

https://github.com/user-attachments/assets/9897f1d6-7b7b-4f9e-8c29-ae674aac5a37



---

### 🔥 Now go flex your eyes. Let your blinks log your legacy.
