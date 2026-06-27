# 🤖 AI Hand Gesture Automation Hub

An AI-powered Hand Gesture Recognition and Computer Automation System built using **Python, OpenCV, MediaPipe, TensorFlow, and Streamlit**. The application recognizes hand gestures in real time using a webcam and performs system actions such as controlling volume, taking screenshots, media control, mouse control, and more.

---

## 📖 Project Overview

AI Hand Gesture Automation Hub enables touch-free interaction with a Windows computer through computer vision and deep learning. The system detects hand landmarks using MediaPipe, classifies gestures using a TensorFlow model, and executes predefined system actions in real time.

This project demonstrates the integration of Computer Vision, Machine Learning, and Automation to create an intuitive Human-Computer Interaction (HCI) system.

---

## ✨ Features

### 🎥 Real-Time Hand Detection
- Detects one or two hands using MediaPipe.
- Tracks 21 hand landmarks.
- Displays hand skeleton with confidence score.
- Live FPS monitoring.

### 🧠 Gesture Recognition
Supports gestures such as:
- 👍 Thumbs Up
- 👎 Thumbs Down
- ✋ Open Palm
- ✊ Fist
- ✌ Peace
- 👌 OK Sign
- ☝ Pointing
- 🤏 Pinch
- 👉 Swipe Right
- 👈 Swipe Left

### 💻 Computer Automation
- 🔊 Increase/Decrease System Volume
- 🎵 Play/Pause Media
- 📸 Capture Screenshots
- 🖱️ Mouse Cursor Control
- 🖱️ Mouse Click
- 📊 PowerPoint Presentation Control
- 🧮 Open Calculator
- 🌐 Launch Browser
- 🔒 Lock Windows
- 💡 Brightness Control
- 🎤 Microphone Mute/Unmute

### 📊 Dashboard
- Live webcam stream
- Detected gesture
- Confidence score
- Current system action
- FPS monitoring
- Gesture history
- System status

### 📁 Logging
- Gesture logs
- Action logs
- Timestamp tracking

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|------------|
| Language | Python |
| Computer Vision | OpenCV |
| Hand Tracking | MediaPipe |
| Deep Learning | TensorFlow / Keras |
| Dashboard | Streamlit |
| Data Processing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Automation | PyAutoGUI, Pycaw |
| Voice Feedback | pyttsx3 |

---

# 📂 Project Structure

```text
AI-Hand-Gesture-Automation/
│
├── dataset/
├── models/
├── logs/
├── screenshots/
├── assets/
│
├── src/
│   ├── collect_data.py
│   ├── train_model.py
│   ├── predict.py
│   ├── hand_detector.py
│   ├── gesture_classifier.py
│   ├── action_controller.py
│   ├── dashboard.py
│   └── utils.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

# ⚙️ Installation

## 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Hand-Gesture-Automation.git

cd AI-Hand-Gesture-Automation
```

---

## 2. Create Virtual Environment

Windows

```bash
python -m venv venv

venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🚀 Usage

## Step 1 — Collect Gesture Dataset

```bash
python src/collect_data.py
```

Collect approximately **200–300 samples** for each gesture.

---

## Step 2 — Train the Model

```bash
python src/train_model.py
```

This generates:

```
models/
├── gesture_model.h5
└── label_encoder.pkl
```

---

## Step 3 — Launch the Dashboard

```bash
streamlit run app.py
```

---

## Step 4 — Start Recognition

Click:

```
🚀 Start Recognition Engine
```

Show gestures to the webcam and observe the corresponding system actions.

---

# 📷 Screenshots

## Dashboard

> Replace the image below with your dashboard screenshot.

```
assets/dashboard.png
```

```markdown
![Dashboard](assets/dashboard.png)
```

---

## Gesture Detection

```
assets/gesture_detection.png
```

```markdown
![Gesture Detection](assets/gesture_detection.png)
```

---

## Volume Control Demo

```
assets/volume_control.png
```

```markdown
![Volume Control](assets/volume_control.png)
```

---

# 🎬 Demo

Add your demo video after uploading it to YouTube or GitHub.

```text
https://youtu.be/YOUR_VIDEO_LINK
```

---

# 🧠 Machine Learning Workflow

```
Webcam
      │
      ▼
OpenCV Video Capture
      │
      ▼
MediaPipe Hand Detection
      │
      ▼
21 Hand Landmark Extraction
      │
      ▼
Feature Preprocessing
      │
      ▼
TensorFlow Gesture Classifier
      │
      ▼
Gesture Prediction
      │
      ▼
System Automation
      │
      ▼
Dashboard Update
```

---

# 📊 Supported Gestures

| Gesture | Action |
|----------|--------|
| 👍 Thumbs Up | Increase Volume |
| 👎 Thumbs Down | Decrease Volume |
| ✋ Open Palm | Play/Pause |
| ✊ Fist | Lock Computer |
| ✌ Peace | Screenshot |
| 👌 OK | Open Calculator |
| ☝ Pointing | Mouse Control |
| 🤏 Pinch | Mouse Click |
| 👉 Swipe Right | Next Slide |
| 👈 Swipe Left | Previous Slide |

---

# 🔮 Future Enhancements

- Sign Language Recognition
- Custom Gesture Training
- Multi-Hand Gesture Recognition
- Gesture-Based Smart Home Control
- Voice Assistant Integration
- Cloud-Based Gesture Analytics
- Cross-Platform Support
- Mobile Application Integration

---

# 👩‍💻 Author

**Vismaya T M**

- GitHub:https://github.com/vismaya418
- LinkedIn: https://www.linkedin.com/in/vismayatm04/

---

# ⭐ Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---
