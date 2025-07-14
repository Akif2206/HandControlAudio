# HandControlAudio
Gesture-based audio controller using MediaPipe, OpenCV, and Pycaw. Control system volume, bass, and frequency using two hands and a webcam — perfect for DJ-style visual interaction or experimental sound control setups.
# Hand Gesture Audio Controller 🎧🖐️

Control system volume, bass, and frequency using hand gestures via webcam using MediaPipe and OpenCV.

### ✨ Features
- 🎚️ Volume: Distance between both index fingers
- 🔊 Bass: Distance between thumb & index of **left hand**
- 🎵 Frequency: Distance between thumb & index of **right hand**
- 🔁 Real-time feedback with animated UI

### 📦 Requirements
pip install -r requirements.txt

> Works on Windows (due to `pycaw` for system volume control)
In your terminal inside the project folder:

bash
Copy
Edit
git init
git add .
git commit -m "Initial commit - Dual hand audio control project"
git remote add origin https://github.com/YOUR_USERNAME/HandControlAudio-DJ.git
git branch -M main
git push -u origin main
