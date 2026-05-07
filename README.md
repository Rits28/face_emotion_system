# Face and Emotion Detection System
**Project P2837554 | Ritika Thapa Magar**

A real-time face detection, emotion recognition, and face identification system built with Python, OpenCV, and dlib. Detected emotions are used to recommend YouTube music via an integrated panel.

---

## Features

- Real-time face detection via Haar cascades
- Emotion recognition — Happy, Sad, Angry, Surprised, Neutral
- Face registration and identification using LBP+HOG features
- YouTube music recommendations based on detected emotion
- Screenshot capture (full frame or face crop)
- Face management — register and delete known faces

---

## Project Structure

```
face_emotion_system/
├── data/
│   └── known_faces/         # Registered face images and feature cache
├── src/
│   ├── __init__.py
│   ├── main.py              # Application entry point
│   ├── face_detector.py     # Haar cascade face detection
│   ├── emotion_estimator.py # Emotion classification (dlib or Haar fallback)
│   ├── face_recognizer.py   # LBP+HOG face identification
│   ├── ui_overlay.py        # Camera feed overlay drawing
│   └── youtube_recommender.py  # Emotion-based YouTube panel
├── tests/
│   ├── __init__.py
│   └── test_system.py
├── shape_predictor_68_face_landmarks.dat  # dlib landmark model (download separately)
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10 or higher
- Windows 10/11 (tested), macOS, or Linux
- Webcam

---

## Installation

### 1. Clone or download the project

```bash
cd Desktop
```

### 2. Install core dependencies

```bash
pip install opencv-python numpy pandas Pillow scipy pytest
```

### 3. Install dlib (optional but recommended)

dlib enables accurate facial landmark-based emotion detection. Without it, the system falls back to Haar cascade emotion estimation.

**Windows:**
```bash
pip install dlib
```

If that fails, install CMake and Visual Studio Build Tools first:
- Download CMake: https://cmake.org/download/
- Download Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/

Then retry:
```bash
pip install dlib
```

**macOS:**
```bash
brew install cmake
pip install dlib
```

**Linux:**
```bash
sudo apt-get install cmake libopenblas-dev liblapack-dev
pip install dlib
```

### 4. Download the dlib landmark model

Download `shape_predictor_68_face_landmarks.dat` from:
https://github.com/davisking/dlib-models

Place the `.dat` file in the **root of the project** (same level as `requirements.txt`):

```
face_emotion_system/
└── shape_predictor_68_face_landmarks.dat  ✓
```

---

## Running the Application

```bash
cd face_emotion_system
python src/main.py
```

---

## How to Use

### Registering a Face
1. Click **"Register New Face"** in the sidebar
2. Enter the person's name
3. Follow the 3-step photo capture — look straight, tilt left, tilt right
4. Click **"Capture Photo"** for each step
5. The face is saved and immediately available for recognition

### Deleting a Face
1. Click **"Delete a Registered Face"**
2. Select the person from the list
3. Confirm deletion — all their photos and feature data are removed

### Taking Screenshots
- Click **"Save Screenshot"** or press **`S`** — saves the full camera frame
- Press **`Shift+S`** — saves a cropped face-only image
- Screenshots are saved in the current working directory

### Keyboard Shortcuts
| Key | Action |
|-----|--------|
| `S` | Save screenshot |
| `Shift+S` | Save face crop |
| `R` | Open register face dialog |
| `Q` | Quit the application |

### YouTube Recommendations
The recommendation panel appears automatically when a face is detected. It updates with music suggestions matching the detected emotion after the emotion is held consistently for ~40 frames.

---

## Emotion Detection

The system uses **dlib 68-point facial landmarks** when available, falling back to Haar cascades otherwise.

| Emotion   | Primary Signal |
|-----------|---------------|
| Happy     | Mouth width ratio increases (MWR > 0.36) |
| Surprised | Eyes widen (EAR > 0.28) + brows raise (BROW > 0.16) |
| Angry     | Brows pulled very low (BROW < 0.11) |
| Sad       | Eyes drooping (EAR < 0.19) |
| Neutral   | All values within baseline range |

---

## Face Recognition

Faces are identified using **LBP (Local Binary Patterns) + HOG (Histogram of Oriented Gradients)** features with chi-squared distance matching.

- Register at least 3 photos per person (straight, left tilt, right tilt) for best accuracy
- Recognition threshold: 13.0 (lower = stricter matching)
- Identity is confirmed after 3 consistent matches in a 12-frame window

---

## Troubleshooting

**Camera not opening:**
```bash
# Try a different camera index in src/main.py
self.cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
```

**numpy DLL error on Windows:**
```bash
pip uninstall numpy opencv-python -y
pip cache purge
pip install numpy
pip install opencv-python
```

**Face always shows "Unknown":**
- Re-register the face in good lighting
- Make sure your face fills the detection box clearly
- Check the terminal for recognizer scores — if scores are well above 13.0, the threshold may need adjusting

**Only Happy/Sad detected (no dlib):**
- Install dlib and download `shape_predictor_68_face_landmarks.dat`
- Place the `.dat` file in the project root folder

---

## Running Tests

```bash
cd face_emotion_system
pytest tests/
```

---

## License

This project was developed for academic purposes.  
Project P2837554 | Ritika Thapa Magar