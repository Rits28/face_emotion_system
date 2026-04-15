# Real-time Face and Emotion Detection System
**Project P2837554 | Ritika Thapa Magar | Niels Brock | Feb 2026**
Supervisor: Bernice Bryan

---

## Overview
A lightweight, real-time prototype that detects faces, recognises known individuals,
and estimates basic emotions — all using classical computer-vision techniques
(OpenCV + rule-based logic). No internet connection or GPU required.

---

## Project Structure
```
face_emotion_system/
├── src/
│   ├── main.py                # Entry point — run this file
│   ├── face_detector.py       # Haar Cascade face detection
│   ├── emotion_estimator.py   # Rule-based emotion estimation
│   ├── face_recognizer.py     # LBP histogram face recognition
│   ├── attendance_logger.py   # SQLite logging + CSV export
│   └── ui_overlay.py          # On-screen annotations / HUD
├── database/
│   └── attendance.db          # Auto-created on first run
├── data/
│   ├── known_faces/           # Add sub-folders here to register faces
│   └── attendance_export.csv  # Auto-created on session end
├── tests/
│   └── test_system.py         # Unit tests (no webcam needed)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the system
```bash
python src/main.py
```

### 3. Keyboard Controls
| Key | Action |
|-----|--------|
| `Q` | Quit and export attendance CSV |
| `S` | Save screenshot |
| `R` | Register a new face (one face must be visible) |

---

## Registering Known Faces

**Method A – Automatic (via webcam)**
1. Run the system.
2. Position your face in the camera.
3. Press `R` and type your name when prompted.

**Method B – Manual (pre-load images)**
1. Create a folder: `data/known_faces/<YourName>/`
2. Place `.jpg` or `.png` photos of the person inside.
3. Restart the system — it will load them automatically.

Example:
```
data/known_faces/
    Alice/
        photo1.jpg
        photo2.jpg
    Bob/
        photo1.jpg
```

---

## How It Works

### Face Detection
Uses OpenCV's `haarcascade_frontalface_default.xml` — a pre-trained Haar Cascade
classifier. The grayscale frame is histogram-equalised first to improve detection
under varied lighting.

### Emotion Estimation (Rule-Based)
Five features are extracted from the face region:
| Feature | Measurement |
|---------|------------|
| Smile score | Haar Cascade smile detection in lower face |
| Eye openness | Eye bounding-box aspect ratio |
| Forehead brightness | Mean pixel intensity, upper 20% of face |
| Edge energy | Laplacian variance in lower face |
| Gradient magnitude | Sobel gradient mean across whole face |

These are combined via hand-tuned thresholds to classify: **Happy, Sad, Angry, Surprised, Neutral**.

### Face Recognition
LBP (Local Binary Pattern) histograms + HSV colour histograms are computed for
each face ROI, then compared against stored features using histogram correlation.
A threshold of **0.55** (out of 1.0) is used to accept a match.

### Attendance & Data Storage
- **SQLite3** database (`database/attendance.db`) stores each recognition event
  with name, emotion, timestamp, date, and session ID.
- On exit the system exports a **CSV** file (`data/attendance_export.csv`) using
  Pandas (falls back to the built-in `csv` module if Pandas is unavailable).

---

## Running the Tests
```bash
pip install pytest
python -m pytest tests/ -v
```
All tests run without a webcam — they use synthetic image data.

---

## Known Limitations
- Detection accuracy depends on lighting quality.
- Rule-based emotion estimation is approximate and works best with clear,
  front-facing expressions.
- Face recognition uses classical features and may confuse people with
  similar skin tones or hairstyles; for higher accuracy consider integrating
  `dlib` / `face-recognition` library (see `requirements.txt`).
- The system is a proof-of-concept prototype, not a production-grade security tool.

---

## References
- OpenCV Documentation: https://docs.opencv.org/
- Dlib Documentation: http://dlib.net/
- Python Software Foundation: https://www.python.org/
- SQLite Documentation: https://www.sqlite.org/
- Zeng et al. (2018) *IEEE TPAMI* 31(1), pp. 39–58.
- Mollahosseini et al. (2016) *AffectNet, CVPR*, pp. 4949–4958.
- Zhang et al. (2016) *IEEE TPAMI* 38(5), pp. 918–930.
