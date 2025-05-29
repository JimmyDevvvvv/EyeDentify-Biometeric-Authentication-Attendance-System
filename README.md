# 👁️ EyeDentify – Blink-Secured Iris Authentication & Attendance System

A lightweight, privacy-aware iris recognition system that uses **blinking** as a liveness check. Built with **PyTorch**, **MediaPipe**, and **OpenCV**, EyeDentify prevents spoofing (e.g., photos) and logs attendance **only when a real, live person is detected**.

---

## 🚀 Features

### 🔑 Core Functionalities
- Iris-based biometric authentication
- Blink detection for spoofing prevention
- CNN-based iris classifier (PyTorch)
- Real-time webcam capture
- CSV-based daily attendance logging
- Clean GUI overlay with live feedback

### 🔐 Anti-Spoofing Measures
- Blink verification (EAR calculation)
- Real-time identity match with liveness check
- Daily de-duplication in attendance log

---

## 🛠️ Technology Stack

| Component     | Role                                |
|---------------|-------------------------------------|
| **PyTorch**   | Iris classification model (CNN)     |
| **MediaPipe** | Facial and eye landmark detection   |
| **OpenCV**    | Webcam feed and image processing    |
| **CSV**       | Persistent attendance logging       |

---

## ⚙️ Setup Instructions

### ✅ Prerequisites
- Python 3.7+
- Webcam device

### 📥 Step 1: Clone the Repository
```bash
git clone https://github.com/your-username/EyeDentify-Biometric-Authentication-Attendance-System.git
cd EyeDentify-Biometric-Authentication-Attendance-System


### 🧪 Step 2: Install Dependencies
