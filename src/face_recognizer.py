import cv2
import numpy as np
import os
import pickle
from collections import deque, Counter


class FaceRecognizer:

    # Scores for this LBP+HOG setup land around 11-12 for the same person.
    # Set threshold to 13.0 to accept known faces and reject strangers.
    THRESHOLD = 13.0
    DEBUG_SCORES = True

    def __init__(self):
        self.known_names    = []
        self.known_features = []

        self._clahe   = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
        self._history = deque(maxlen=12)

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
                names    = data["names"]
                features = data["features"]

                if features:
                    cached_dim = len(features[0])
                    probe_dim  = self._probe_feature_dim()
                    if probe_dim is not None and cached_dim != probe_dim:
                        print(f"  [WARN] Cache dim {cached_dim} != expected {probe_dim}. Rebuilding...")
                        os.remove(features_file)
                    else:
                        self.known_names    = names
                        self.known_features = features
                        print(f"  [OK] Loaded {len(self.known_names)} face samples from cache.")
                        return
                else:
                    self.known_names    = names
                    self.known_features = features
                    return
            except Exception:
                pass

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
            print(f"  [OK] Built feature cache for {len(set(self.known_names))} people.")

    def _save_features(self, directory: str):
        pkl = os.path.join(directory, "features.pkl")
        data = {
            "names":    self.known_names,
            "features": self.known_features,
        }
        with open(pkl, "wb") as f:
            pickle.dump(data, f)

    def _probe_feature_dim(self):
        try:
            blank = np.zeros((80, 80, 3), dtype=np.uint8)
            feat  = self._extract_features(blank)
            return len(feat) if feat is not None else None
        except Exception:
            return None

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

        query_dim = len(query)

        dist_per_person: dict[str, list[float]] = {}
        for name, feat in zip(self.known_names, self.known_features):
            if len(feat) != query_dim:
                continue
            d = self._chi2_distance(query, feat)
            dist_per_person.setdefault(name, []).append(d)

        if not dist_per_person:
            return "Unknown"

        best_name  = None
        best_score = float("inf")
        for name, dists in dist_per_person.items():
            top3  = sorted(dists)[:3]
            score = sum(top3) / len(top3)
            if score < best_score:
                best_score = score
                best_name  = name

        if self.DEBUG_SCORES:
            all_scores = {
                n: round(sum(sorted(d)[:3]) / len(sorted(d)[:3]), 4)
                for n, d in dist_per_person.items()
            }
            print(
                f"  [Recognizer] scores={all_scores}  "
                f"best={best_name} ({best_score:.4f})  threshold={self.THRESHOLD}"
            )

        return best_name if best_score <= self.THRESHOLD else "Unknown"

    def recognize_stable(self, face_roi: np.ndarray) -> str:
        name = self.recognize(face_roi)
        self._history.append(name)

        most_common, count = Counter(self._history).most_common(1)[0]
        if most_common != "Unknown" and count >= 3:
            return most_common
        return "Unknown"

    def _extract_features(self, face_roi: np.ndarray) -> np.ndarray | None:
        if face_roi is None or face_roi.size == 0:
            return None

        resized = cv2.resize(face_roi, (80, 80))
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        gray    = self._clahe.apply(gray)

        h, w = gray.shape

        rows, cols = 5, 5
        cell_h = h // rows
        cell_w = w // cols
        lbp_parts = []
        for r in range(rows):
            for c in range(cols):
                cell = gray[r*cell_h:(r+1)*cell_h, c*cell_w:(c+1)*cell_w]
                lbp_parts.append(self._lbp_histogram(cell))
        lbp_feat = np.concatenate(lbp_parts)

        gx  = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy  = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx**2 + gy**2)
        ang = (np.arctan2(gy, gx) * 180.0 / np.pi) % 180.0

        hog_parts = []
        hog_rows, hog_cols = 4, 4
        hog_h = h // hog_rows
        hog_w = w // hog_cols
        for r in range(hog_rows):
            for c in range(hog_cols):
                m = mag[r*hog_h:(r+1)*hog_h, c*hog_w:(c+1)*hog_w]
                a = ang[r*hog_h:(r+1)*hog_h, c*hog_w:(c+1)*hog_w]
                hist, _ = np.histogram(a, bins=9, range=(0, 180), weights=m)
                hist = hist / (hist.sum() + 1e-7)
                hog_parts.append(hist)
        hog_feat = np.concatenate(hog_parts)

        return np.concatenate([lbp_feat, hog_feat]).astype(np.float32)

    def _lbp_histogram(self, gray: np.ndarray) -> np.ndarray:
        if gray.shape[0] < 3 or gray.shape[1] < 3:
            return np.zeros(256, dtype=np.float32)

        c = gray[1:-1, 1:-1].astype(np.int16)
        neighbors = [
            gray[0:-2, 0:-2], gray[0:-2, 1:-1], gray[0:-2, 2:],
            gray[1:-1, 2:],   gray[2:,   2:],   gray[2:,   1:-1],
            gray[2:,   0:-2], gray[1:-1, 0:-2],
        ]
        lbp = np.zeros_like(c, dtype=np.uint8)
        for bit, nb in enumerate(neighbors):
            lbp |= ((nb.astype(np.int16) >= c) << bit).astype(np.uint8)

        hist = cv2.calcHist([lbp], [0], None, [256], [0, 256]).flatten()
        return hist / (hist.sum() + 1e-7)

    def _chi2_distance(self, h1: np.ndarray, h2: np.ndarray) -> float:
        num = (h1 - h2) ** 2
        den = h1 + h2 + 1e-10
        return float(0.5 * np.sum(num / den))