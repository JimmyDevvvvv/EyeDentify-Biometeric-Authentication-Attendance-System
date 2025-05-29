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
🧪 Step 2: Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
If you don’t have a requirements.txt file, you can manually install:

bash
Copy
Edit
pip install opencv-python mediapipe torch torchvision numpy
🚀 Usage
👤 Step 1: Enroll a New User
bash
Copy
Edit
python dataset_capture.py
You'll be prompted:

yaml
Copy
Edit
Enter username:
➡️ Enter a unique name like ahmed, sarah, or jimmy.
Then:

Sit in front of the webcam

Center your face

System will capture 30 eye samples

Folder dataset/{username} will be created

Repeat this step for each person you want to enroll.

🧠 Step 2: Train the Model
bash
Copy
Edit
python train_model.py
This will:

Load all folders under dataset/

Train a CNN model to recognize each user

Save the trained model as:

bash
Copy
Edit
model/iris_model.pth
Automatically generate the class map:

python
Copy
Edit
{0: 'jimmy', 1: 'ahmed', 2: 'sarah'}
👁️ Step 3: Run the Blink-Secured Scanner
bash
Copy
Edit
python predict_live_with_blink.py
Then:

The webcam launches

User must blink to prove liveness

If blink and identity match:

yaml
Copy
Edit
✅ Authenticated: sarah
If no blink:

Copy
Edit
⚠️ Blink Required
If not in the dataset:

mathematica
Copy
Edit
❌ Not recognized
📝 Step 4: Attendance Logging
Once a user is successfully authenticated with a blink, a line is logged in attendance_log.csv:

yaml
Copy
Edit
sarah, 2025-05-29 09:14 AM
✅ Only one log entry per person per day

🧼 De-duplication is handled automatically

📂 Folder Structure
bash
Copy
Edit
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
└── requirements.txt
🧠 How It Works
Component	Function
MediaPipe	Facial & eye landmark detection
OpenCV	Webcam access + image processing
PyTorch	CNN for iris image classification
EAR (Eye Ratio)	Used to detect blinking (anti-spoofing)
CSV Logger	Timestamp-based attendance tracker

🌱 Future Work Ideas
🔐 Add Homomorphic Encryption to protect iris data

🕵️ Add Zero-Knowledge Proofs for anonymous verification

📊 Add a browser-based dashboard to view logs

🔄 Add challenge-response blinking (e.g., blink twice)

🤖 Use transfer learning with more complex CNN architectures

👨‍💻 Author
Built with 💻 by Mohamed Gamal(Jimmy)
 
