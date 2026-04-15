"""
face_recognizer.py - Face recognition using LBP + HSV features.
Threshold raised for accuracy, supports multiple samples per person.
"""
import cv2
import numpy as np
import os
import pickle


class FaceRecognizer:

    THRESHOLD = 0.72  # above this = recognised

    def __init__(self):
        self.known_names    = []
        self.known_features = []

    def load_known_faces(self, directory: str):
        self.known_names    = []
        self.known_features = []

        if not os.path.isdir(directory):
            return

        features_file = os.path.join(directory, "features.pkl")
        if os.path.exists(features_file):
            try:
                with open(features_file, "rb") as f:
                    data = pickle.load(f)
                    self.known_names    = data["names"]
                    self.known_features = data["features"]
                return
            except Exception:
                pass  # corrupt cache — rebuild below

        for person_name in sorted(os.listdir(directory)):
            person_dir = os.path.join(directory, person_name)
            if not os.path.isdir(person_dir):
                continue
            for img_file in sorted(os.listdir(person_dir)):
                if not img_file.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                img = cv2.imread(os.path.join(person_dir, img_file))
                if img is None:
                    continue
                feat = self._extract_features(img)
                if feat is not None:
                    self.known_names.append(person_name)
                    self.known_features.append(feat)

        if self.known_names:
            self._save_features(directory)

    def register_face(self, frame: np.ndarray, bbox: tuple,
                      name: str, directory: str):
        x, y, w, h = bbox
        face_roi   = frame[y:y+h, x:x+w]

        person_dir = os.path.join(directory, name)
        os.makedirs(person_dir, exist_ok=True)
        existing   = [f for f in os.listdir(person_dir) if f.endswith(".jpg")]
        img_path   = os.path.join(person_dir, f"face_{len(existing)+1:03d}.jpg")
        cv2.imwrite(img_path, face_roi)

        feat = self._extract_features(face_roi)
        if feat is not None:
            self.known_names.append(name)
            self.known_features.append(feat)

        # Clear cache
        pkl = os.path.join(directory, "features.pkl")
        if os.path.exists(pkl):
            os.remove(pkl)
        self._save_features(directory)

    def recognize(self, face_roi: np.ndarray) -> str:
        if not self.known_features or face_roi.size == 0:
            return "Unknown"

        query = self._extract_features(face_roi)
        if query is None:
            return "Unknown"

        # Score against every stored sample, group by name
        scores = {}
        for name, feat in zip(self.known_names, self.known_features):
            s = self._compare(query, feat)
            if name not in scores or s > scores[name]:
                scores[name] = s

        best_name  = max(scores, key=scores.get)
        best_score = scores[best_name]

        return best_name if best_score >= self.THRESHOLD else "Unknown"

    def _extract_features(self, face_roi):
        if face_roi is None or face_roi.size == 0:
            return None
        resized  = cv2.resize(face_roi, (80, 80))  # slightly larger for better LBP
        gray     = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        lbp_hist = self._lbp_histogram(gray)
        hsv      = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
        h_hist   = cv2.calcHist([hsv], [0], None, [24], [0, 180]).flatten()
        s_hist   = cv2.calcHist([hsv], [1], None, [24], [0, 256]).flatten()
        h_hist  /= (h_hist.sum() + 1e-7)
        s_hist  /= (s_hist.sum() + 1e-7)
        return np.concatenate([lbp_hist, h_hist, s_hist]).astype(np.float32)

    def _lbp_histogram(self, gray: np.ndarray) -> np.ndarray:
        rows, cols = gray.shape
        lbp        = np.zeros_like(gray, dtype=np.uint8)
        offsets    = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        for i in range(1, rows-1):
            for j in range(1, cols-1):
                center = int(gray[i, j])
                code   = 0
                for bit, (di, dj) in enumerate(offsets):
                    if int(gray[i+di, j+dj]) >= center:
                        code |= (1 << bit)
                lbp[i, j] = code
        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
        return hist / (hist.sum() + 1e-7)

    @staticmethod
    def _compare(a, b) -> float:
        score = cv2.compareHist(
            a.reshape(-1, 1), b.reshape(-1, 1), cv2.HISTCMP_CORREL)
        return float((score + 1.0) / 2.0)

    def _save_features(self, directory: str):
        pkl = os.path.join(directory, "features.pkl")
        with open(pkl, "wb") as f:
            pickle.dump({"names": self.known_names,
                         "features": self.known_features}, f)