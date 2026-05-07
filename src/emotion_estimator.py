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
    "Sad":       (200,  80,  50),
    "Angry":     (0,    40, 220),
    "Surprised": (0,   200, 200),
    "Neutral":   (200, 200, 200),
    "Unknown":   (128, 128, 128),
}

LANDMARK_MODEL = "shape_predictor_68_face_landmarks.dat"


class EmotionEstimator:

    def __init__(self):
        self._history = deque(maxlen=8)

        if DLIB_AVAILABLE:
            try:
                self.predictor = dlib.shape_predictor(LANDMARK_MODEL)
                self._mode     = "dlib"
                print("  [OK] Emotion estimator using dlib landmarks")
            except Exception:
                self._mode = "haar"
                print("  [WARN] .dat file not found - using Haar fallback")
        else:
            self._mode = "haar"
            print("  [WARN] dlib not installed - using Haar fallback")

        self.smile_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_smile.xml")
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")

        self.DEBUG = False

    def estimate(self, frame: np.ndarray, bbox: tuple):
        x, y, w, h = bbox
        if frame[y:y+h, x:x+w].size == 0:
            return "Unknown", 0.0

        if self._mode == "dlib":
            emotion, conf = self._estimate_dlib(frame, bbox)
        else:
            emotion, conf = self._estimate_haar(frame, bbox)

        if conf >= 0.55:
            self._history.append(emotion)

        if self._history:
            smooth = Counter(self._history).most_common(1)[0][0]
        else:
            smooth = emotion

        return smooth, conf

    # DLIB PATH

    def _estimate_dlib(self, frame, bbox):
        x, y, w, h = bbox
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        rect  = dlib.rectangle(x, y, x + w, y + h)
        shape = self.predictor(gray, rect)

        pts = np.array([[shape.part(i).x, shape.part(i).y]
                        for i in range(68)], dtype=np.float32)

        ear  = self._eye_aspect_ratio(pts)
        mar  = self._mouth_aspect_ratio(pts)
        mwr  = self._mouth_width_ratio(pts, w)
        brow = self._brow_raise_ratio(pts, h)

        mca_left  = (pts[48][1] - pts[51][1]) / (np.linalg.norm(pts[48] - pts[51]) + 1e-6)
        mca_right = (pts[54][1] - pts[51][1]) / (np.linalg.norm(pts[54] - pts[51]) + 1e-6)
        ibd       = np.linalg.norm(pts[21] - pts[22]) / w

        if self.DEBUG:
            print(
                f"EAR:{ear:.3f} MAR:{mar:.3f} MWR:{mwr:.3f} "
                f"BROW:{brow:.3f} MCA_L:{mca_left:.3f} "
                f"MCA_R:{mca_right:.3f} IBD:{ibd:.3f}"
            )

        return self._classify(ear, mar, mwr, brow, mca_left, mca_right, ibd)

    def _classify(self, ear, mar, mwr, brow, mca_left, mca_right, ibd):
        """
        Tuned thresholds from real live data:

        HAPPY     MWR > 0.36  — mouth clearly widens when smiling
                  BROW > 0.16 — brows slightly raised with smile

        SURPRISED EAR > 0.28  — eyes noticeably wider than neutral (~0.24)
                  BROW > 0.16 — brows raised (distinguishes from neutral wide eyes)

        ANGRY     BROW < 0.11 — brows pulled very low/furrowed
                  (neutral BROW is ~0.14, angry drops to ~0.10 or below)

        SAD       EAR < 0.19  — eyes drooping/heavy
                  (neutral EAR is ~0.24, sad drops to ~0.17)
                  MCA values overlap too much with neutral to use alone

        NEUTRAL   everything else — EAR 0.19-0.28, MWR < 0.36, BROW 0.11-0.16
        """

        # Happy: mouth widens + brows slightly raised 
        if mwr > 0.36 and brow > 0.15:
            conf = self._conf([mwr > 0.36, mwr > 0.38, brow > 0.16])
            return "Happy", conf

        # Happy fallback: very wide mouth alone 
        if mwr > 0.38:
            return "Happy", 0.70

        # Surprised: eyes wider than normal + brows raised 
        if ear > 0.28 and brow > 0.16:
            conf = self._conf([ear > 0.28, ear > 0.30, brow > 0.16, brow > 0.17])
            return "Surprised", conf

        # Surprised fallback: very wide eyes alone 
        if ear > 0.32:
            return "Surprised", 0.65

        # Angry: brows furrowed very low 
        if brow < 0.11:
            conf = self._conf([brow < 0.11, brow < 0.10, ibd < 0.13])
            return "Angry", conf

        #  Angry fallback: narrow brow gap + low brows 
        if ibd < 0.12 and brow < 0.13:
            return "Angry", 0.65

        #  Sad: eyes drooping (primary signal for this face) 
        if ear < 0.19:
            conf = self._conf([ear < 0.19, ear < 0.18, ear < 0.17])
            return "Sad", conf

        #  Sad fallback: slightly droopy eyes + downturned corners 
        if ear < 0.21 and (mca_right > 0.35 or mca_left > 0.32):
            return "Sad", 0.62

        #  Neutral ---
        return "Neutral", 0.80

    # DLIB HELPERS

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
        return (A + B + C) / (3.0 * D + 1e-6)

    def _mouth_width_ratio(self, pts, face_w):
        return dist.euclidean(pts[48], pts[54]) / (face_w + 1e-6)

    def _brow_raise_ratio(self, pts, face_h):
        left_brow   = np.mean(pts[17:22, 1])
        right_brow  = np.mean(pts[22:27, 1])
        left_eye    = np.mean(pts[37:39, 1])
        right_eye   = np.mean(pts[43:45, 1])
        left_raise  = (left_eye  - left_brow)  / (face_h + 1e-6)
        right_raise = (right_eye - right_brow) / (face_h + 1e-6)
        return (left_raise + right_raise) / 2.0

    # HAAR FALLBACK PATH

    def _estimate_haar(self, frame, bbox):
        x, y, w, h = bbox
        face_roi  = frame[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        gray_face = cv2.equalizeHist(gray_face)

        eye_region = gray_face[int(h * 0.15):int(h * 0.50), :]
        eyes       = self.eye_cascade.detectMultiScale(
            eye_region, scaleFactor=1.1, minNeighbors=5, minSize=(15, 15))
        num_eyes = len(eyes)

        gx       = cv2.Sobel(gray_face, cv2.CV_32F, 1, 0, ksize=3)
        gy       = cv2.Sobel(gray_face, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = float(np.mean(np.sqrt(gx**2 + gy**2))) / 128.0

        brow_region = gray_face[int(h * 0.10):int(h * 0.30), :]
        brow_mean   = float(np.mean(brow_region)) / 255.0

        lower       = gray_face[int(h * 0.50):h, :]
        edge_energy = float(np.mean(np.abs(cv2.Laplacian(lower, cv2.CV_64F)))) / 50.0

        forehead   = float(np.mean(gray_face[0:int(h * 0.20), :])) / 255.0

        mouth_crop = gray_face[int(h * 0.60):h, int(w * 0.20):int(w * 0.80)]
        mouth_var  = float(np.std(mouth_crop))
        mouth_top  = float(np.mean(gray_face[int(h * 0.58):int(h * 0.68), int(w*0.25):int(w*0.75)]))
        mouth_bot  = float(np.mean(gray_face[int(h * 0.78):int(h * 0.92), int(w*0.25):int(w*0.75)]))
        mouth_open = abs(mouth_top - mouth_bot) / 255.0

        if self.DEBUG:
            print(
                f"eyes={num_eyes} grad={grad_mag:.2f} brow={brow_mean:.2f} "
                f"edge={edge_energy:.2f} forehead={forehead:.2f} "
                f"mouth_var={mouth_var:.1f} mouth_open={mouth_open:.3f}"
            )

        if mouth_var > 10 and mouth_open > 0.04:
            return "Surprised", round(min(0.60 + mouth_open, 0.92), 2)
        if mouth_var > 14:
            return "Surprised", 0.65

        mouth_region = gray_face[int(h * 0.55):h, :]
        smiles = self.smile_cascade.detectMultiScale(
            mouth_region, scaleFactor=1.5, minNeighbors=15, minSize=(20, 10))
        if len(smiles) > 0:
            return "Happy", round(0.65 + min(len(smiles), 3) * 0.10, 2)

        if brow_mean < 0.38 and grad_mag > 0.45:
            return "Angry", round(min(0.55 + grad_mag * 0.3, 0.90), 2)
        if edge_energy > 0.65 and grad_mag > 0.55:
            return "Angry", round(min((edge_energy + grad_mag) / 2.0, 0.90), 2)

        if grad_mag < 0.32 and forehead < 0.42:
            return "Sad", round(0.50 + (0.42 - forehead), 2)
        if grad_mag < 0.25:
            return "Sad", 0.60

        return "Neutral", round(max(0.55, 1.0 - grad_mag * 0.4), 2)

    # SHARED HELPERS
    

    def _conf(self, conditions):
        met = sum(conditions)
        return round(0.60 + (met / len(conditions)) * 0.35, 2)