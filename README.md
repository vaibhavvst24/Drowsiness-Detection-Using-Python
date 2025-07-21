# 💤 Drowsiness Detection using Python & Machine Learning

This project is a real-time drowsiness detection system developed in Python. It uses computer vision to detect eye landmarks, calculates the Eye Aspect Ratio (EAR), and predicts drowsiness using a trained machine learning model. When drowsiness is detected, an alarm is triggered to alert the user.

---

## 🎯 Features

- Real-time video processing using webcam
- Facial landmark detection using **MediaPipe**
- Eye Aspect Ratio (EAR) calculation
- Rolling statistical features (mean, std) for smoothing
- Trained **Random Forest** model for drowsiness prediction
- Audio alert system using **Pygame**
- Modular, readable, and efficient Python code

---

## 🧠 Tech Stack

- **Python 3**
- **OpenCV** – for real-time video capture and rendering
- **MediaPipe** – to detect facial and eye landmarks
- **Scikit-learn** – for model training and evaluation
- **Pandas & NumPy** – for data manipulation
- **Joblib** – to save and load trained ML models
- **Pygame** – for playing the drowsiness alarm
- **Threading** – to manage non-blocking audio alerts

---

## 📦 Project Structure

Drowsiness-Detection-using-python/
├── Main.py # Real-time detection script
├── alarm.wav # Alarm sound file
├── requirements.txt # Python dependencies
└── README.md # This file


---

## 🛠️ Setup Instructions

### 🔗 Prerequisites

- Python 3.7+
- Webcam (built-in or USB)
- `alarm.wav` file for alert sound

## 🚀 Running the App

python Main.py
