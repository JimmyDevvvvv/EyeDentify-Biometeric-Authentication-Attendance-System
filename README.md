# EyeDentify-Biometeric-Authentication-Attendance-System


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

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/eye-dentify.git
cd eye-dentify













🚀 Usage
✅ Step 1: Enroll a New User
bash
Copy
Edit
python dataset_capture.py
You’ll be prompted:

yaml
Copy
Edit
Enter username:
➡️ Type a unique name (e.g., ahmed, sarah, nour).

Then:

Center the user's face

The system captures 30 iris shots

The folder dataset/{username} will be created automatically

Repeat this step for each new person you want to register.

✅ Step 2: Train the Model
After enrolling users, train the model using:

bash
Copy
Edit
python train_model.py
This will:

Read all folders under dataset/

Assign each folder a class label

Train a CNN to classify iris patterns

Save the model to: model/iris_model.pth

You’ll also see the mapping (label → username):

python
Copy
Edit
{0: "jimmy", 1: "ahmed", 2: "sarah"}
✅ Step 3: Run the Live Authentication Scanner
bash
Copy
Edit
python predict_live_with_blink.py
This script will:

Open the webcam

Detect a user’s eyes and facial landmarks

Check for a blink (via EAR – Eye Aspect Ratio)

Crop the eye region and pass it to the model

If a match is found with ≥ 80% confidence and a blink is detected:

yaml
Copy
Edit
✅ Authenticated: sarah
If the user is not enrolled:

mathematica
Copy
Edit
❌ Not recognized
If the user doesn’t blink (spoof/photo):

Copy
Edit
⚠️ Blink Required
✅ Step 4: Attendance Logging
Each successfully authenticated user is logged in:

Copy
Edit
attendance_log.csv
Example entry:

yaml
Copy
Edit
sarah, 2025-05-29 09:14 AM
✨ Smart logging: each user is only logged once per day.

🗂️ Folder Structure
bash
Copy
Edit
.
├── dataset/                      # User eye image folders (auto-created)
│   ├── ahmed/
│   ├── sarah/
│   └── ...
├── model/
│   └── iris_model.pth           # Saved PyTorch model
├── utils/
│   └── preprocess.py            # Preprocessing logic
├── attendance_log.csv           # Attendance records (auto-generated)
├── dataset_capture.py           # Tool to collect user images
├── train_model.py               # Model training script
├── predict_live_with_blink.py  # Liveness + authentication engine
├── README.md
└── requirements.txt
💡 Tech Highlights
Component	Tool
Face Detection	MediaPipe
Blink Detection	Eye Aspect Ratio (EAR)
Model	PyTorch CNN
Image I/O	OpenCV
Liveness Logic	Blink + Softmax Confidence
Logs	CSV

🔮 Future Enhancements
🧬 Add Homomorphic Encryption to dataset storage

🕵️ Implement Zero-Knowledge Proofs for attendance verification without identity exposure

🖥️ Launch a visual dashboard for admin to monitor attendance in real time

🔐 Challenge-response based anti-spoofing

📡 Remote/IoT camera integration

💬 Author
Built by Jimmy (a.k.a Mr. Robot)
📍 Cybersecurity + ML Developer
🛡️ Focused on privacy, authentication, and anti-spoofing solutions

Guided by: Connor, your AI wingman 🤖








