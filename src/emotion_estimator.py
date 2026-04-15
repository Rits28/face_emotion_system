import cv2
import numpy as np
from collections import deque, Counter

try:
    import dlib
    from scipy.spatial import distance as dist
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False

EMOTION_COLORS = {
    "Happy":     (0,   220,  80),
    "Sad":       (200,  80,   0),
    "Angry":     (0,    40, 220),
    "Surprised": (0,   200, 200),
    "Excited":   (0,   255, 255),
    "Neutral":   (200, 200, 200),
    "Unknown":   (128, 128, 128),
}

LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"


class EmotionEstimator:
    def __init__(self):
        self._history = deque(maxlen=6)  # reduced from 10 — faster response

        if DLIB_AVAILABLE:
            try:
                self.predictor = dlib.shape_predictor(LANDMARK_MODEL)
                self._mode = "dlib"
                print("  [OK] Emotion estimator using dlib landmarks")
            except Exception:
                self._mode = "haar"
                print("  [WARN] landmark .dat not found — falling back to Haar rules")
        else:
            self._mode = "haar"
            print("  [WARN] dlib not installed — using Haar rule-based estimator")

        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )

    def estimate(self, frame: np.ndarray, bbox: tuple):
        x, y, w, h = bbox
        if frame[y:y+h, x:x+w].size == 0:
            return "Unknown", 0.0

        if self._mode == "dlib":
            emotion, conf = self._estimate_dlib(frame, bbox)
        else:
            emotion, conf = self._estimate_haar(frame, bbox)

        self._history.append(emotion)
        smooth = Counter(self._history).most_common(1)[0][0]
        return smooth, conf

    def _estimate_dlib(self, frame, bbox):
        x, y, w, h = bbox
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray  = cv2.equalizeHist(gray)
        rect  = dlib.rectangle(x, y, x + w, y + h)
        shape = self.predictor(gray, rect)
        pts   = np.array([[shape.part(i).x, shape.part(i).y]
                          for i in range(68)], dtype=np.float32)

        ear  = self._eye_aspect_ratio(pts)
        mar  = self._mouth_aspect_ratio(pts)
        mwr  = self._mouth_width_ratio(pts, w)
        brow = self._brow_raise_ratio(pts, h)

        # Uncomment to tune thresholds for your face:
        # print(f"EAR:{ear:.2f} MAR:{mar:.2f} MWR:{mwr:.2f} BROW:{brow:.2f}")

        return self._classify_landmarks(ear, mar, mwr, brow)

    def _classify_landmarks(self, ear, mar, mwr, brow):
        print(f"EAR:{ear:.2f} MAR:{mar:.2f} MWR:{mwr:.2f} BROW:{brow:.2f}")
        # Excited: wide eyes + open mouth + raised brows
        if ear > 0.28 and mar > 0.12 and brow > 0.22:
            return "Excited",   self._conf([ear>0.28, mar>0.12, brow>0.22])

        # Happy: wide smile
        if mwr > 0.42:
            return "Happy",     self._conf([mwr>0.42, ear>0.20])

        # Surprised: wide eyes + raised brows
        if ear > 0.30 and brow > 0.22:
            return "Surprised", self._conf([ear>0.30, brow>0.22])

        # Angry: low brows + tight mouth
        if brow < 0.18 and mar < 0.10:
            return "Angry",     self._conf([brow<0.18, mar<0.10])

        # Sad: low brows + slightly closed eyes + narrow mouth
        if brow < 0.20 and ear < 0.28 and mwr < 0.44:
            return "Sad",       self._conf([brow<0.20, ear<0.28])

        # Neutral
        return "Neutral", 0.75

    def _eye_aspect_ratio(self, pts):
        def ear(eye):
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            return (A + B) / (2.0 * C)
        return (ear(pts[36:42]) + ear(pts[42:48])) / 2.0

    def _mouth_aspect_ratio(self, pts):
        A = dist.euclidean(pts[51], pts[59])
        B = dist.euclidean(pts[52], pts[58])
        C = dist.euclidean(pts[53], pts[57])
        D = dist.euclidean(pts[48], pts[54])
        return (A + B + C) / (3.0 * D)

    def _mouth_width_ratio(self, pts, face_w):
        return dist.euclidean(pts[48], pts[54]) / face_w

    def _brow_raise_ratio(self, pts, face_h):
        left_brow   = np.mean(pts[17:22, 1])
        right_brow  = np.mean(pts[22:27, 1])
        left_eye    = np.mean(pts[37:39, 1])
        right_eye   = np.mean(pts[43:45, 1])
        left_raise  = (left_eye  - left_brow)  / face_h
        right_raise = (right_eye - right_brow) / face_h
        return (left_raise + right_raise) / 2.0

    def _estimate_haar(self, frame, bbox):
        x, y, w, h = bbox
        face_roi   = frame[y:y+h, x:x+w]
        gray_face  = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_face  = cv2.equalizeHist(gray_face)

        mouth_region = gray_face[int(h*0.55):h, :]
        smiles = self.smile_cascade.detectMultiScale(
            mouth_region, scaleFactor=1.7, minNeighbors=18, minSize=(25, 15))
        eye_region = gray_face[int(h*0.15):int(h*0.50), :]
        eyes = self.eye_cascade.detectMultiScale(
            eye_region, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        lower       = gray_face[int(h*0.50):h, :]
        edge_energy = float(np.mean(np.abs(cv2.Laplacian(lower, cv2.CV_64F)))) / 50.0
        gx          = cv2.Sobel(gray_face, cv2.CV_32F, 1, 0, ksize=3)
        gy          = cv2.Sobel(gray_face, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag    = float(np.mean(np.sqrt(gx**2 + gy**2))) / 128.0
        forehead    = np.mean(gray_face[0:int(h*0.20), :]) / 255.0

        if len(smiles) > 0:
            return "Happy",     round(0.6 + min(len(smiles), 2) * 0.15, 2)
        if len(eyes) > 2 and forehead > 0.55:
            return "Surprised", 0.65
        if edge_energy > 0.55 and grad_mag > 0.55:
            return "Angry",     round(min((edge_energy + grad_mag) / 2.0, 1.0), 2)
        if grad_mag < 0.30 and forehead < 0.45:
            return "Sad",       round(0.45 + (0.45 - grad_mag), 2)
        return "Neutral",       round(1.0 - grad_mag * 0.5, 2)

    def _conf(self, conditions):
        met = sum(conditions)
        return round(0.60 + (met / len(conditions)) * 0.35, 2)

    @staticmethod
    def get_color(emotion: str) -> tuple:
        return EMOTION_COLORS.get(emotion, (200, 200, 200))