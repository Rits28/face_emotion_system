import sys
import os
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.face_detector import FaceDetector
from src.emotion_estimator import EmotionEstimator
from src.face_recognizer import FaceRecognizer
from src.attendance_logger import AttendanceLogger



# Helpers

def make_blank_frame(h=480, w=640, c=3):
    """Create a plain grey BGR frame."""
    return np.full((h, w, c), 128, dtype=np.uint8)

def make_face_frame():
    """Create a frame with a synthetic white rectangle (simulating a face ROI)."""
    frame = make_blank_frame()
    # Draw a white region to simulate a face
    frame[100:220, 200:320] = 200
    return frame



# FaceDetector tests


class TestFaceDetector:
    def test_init(self):
        fd = FaceDetector()
        assert fd.cascade is not None
        assert not fd.cascade.empty()

    def test_detect_returns_list(self):
        fd = FaceDetector()
        result = fd.detect(make_blank_frame())
        assert isinstance(result, list)

    def test_detect_no_crash_on_empty_frame(self):
        fd = FaceDetector()
        tiny = np.zeros((10, 10, 3), dtype=np.uint8)
        result = fd.detect(tiny)
        assert isinstance(result, list)

    def test_approximate_landmarks(self):
        fd = FaceDetector()
        pts = fd._approximate_landmarks(make_blank_frame(), (100, 100, 120, 120))
        assert len(pts) == 6
        for (px, py) in pts:
            assert isinstance(px, int)
            assert isinstance(py, int)



# EmotionEstimator tests

class TestEmotionEstimator:
    EMOTIONS = {"Happy", "Sad", "Angry", "Surprised", "Neutral", "Unknown"}

    def test_init(self):
        ee = EmotionEstimator()
        assert ee.smile_cascade is not None

    def test_estimate_returns_tuple(self):
        ee = EmotionEstimator()
        frame = make_face_frame()
        emotion, conf = ee.estimate(frame, (200, 100, 120, 120))
        assert emotion in self.EMOTIONS
        assert 0.0 <= conf <= 1.0

    def test_estimate_empty_bbox(self):
        ee = EmotionEstimator()
        frame = make_blank_frame()
        emotion, conf = ee.estimate(frame, (0, 0, 0, 0))
        assert emotion == "Unknown"

    def test_classify_happy(self):
        ee = EmotionEstimator()
        label, conf = ee._classify({"smile": 0.9, "eye_ratio": 0.4,
                                     "forehead_bright": 0.6, "edge_energy": 0.3,
                                     "gradient_mag": 0.5})
        assert label == "Happy"

    def test_classify_surprised(self):
        ee = EmotionEstimator()
        label, _ = ee._classify({"smile": 0.0, "eye_ratio": 0.65,
                                  "forehead_bright": 0.70, "edge_energy": 0.3,
                                  "gradient_mag": 0.4})
        assert label == "Surprised"

    def test_classify_angry(self):
        ee = EmotionEstimator()
        label, _ = ee._classify({"smile": 0.0, "eye_ratio": 0.35,
                                  "forehead_bright": 0.5, "edge_energy": 0.7,
                                  "gradient_mag": 0.7})
        assert label == "Angry"

    def test_classify_neutral(self):
        ee = EmotionEstimator()
        label, _ = ee._classify({"smile": 0.0, "eye_ratio": 0.35,
                                  "forehead_bright": 0.55, "edge_energy": 0.3,
                                  "gradient_mag": 0.3})
        assert label == "Neutral"

    def test_get_color_returns_tuple(self):
        color = EmotionEstimator.get_color("Happy")
        assert isinstance(color, tuple)
        assert len(color) == 3


# FaceRecognizer tests


class TestFaceRecognizer:
    def test_init(self):
        fr = FaceRecognizer()
        assert fr.known_names == []
        assert fr.known_features == []

    def test_recognize_no_known_faces(self):
        fr = FaceRecognizer()
        face = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        assert fr.recognize(face) == "Unknown"

    def test_recognize_after_register(self, tmp_path):
        fr = FaceRecognizer()
        # Create a synthetic face image
        face_img = np.random.randint(100, 200, (100, 100, 3), dtype=np.uint8)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        frame[100:200, 100:200] = face_img

        fr.register_face(frame, (100, 100, 100, 100), "TestPerson", str(tmp_path))
        assert "TestPerson" in fr.known_names

    def test_feature_extraction_shape(self):
        fr = FaceRecognizer()
        face = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        feat = fr._extract_features(face)
        assert feat is not None
        assert feat.ndim == 1
        assert feat.shape[0] > 0

    def test_compare_identical_features(self):
        fr = FaceRecognizer()
        face = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        feat = fr._extract_features(face)
        score = fr._compare(feat, feat)
        assert score > 0.95  # Same feature → high similarity


# AttendanceLogger tests

class TestAttendanceLogger:
    def test_init(self, tmp_path):
        db = str(tmp_path / "test.db")
        logger = AttendanceLogger(db_path=db)
        assert os.path.exists(db)

    def test_log_entry(self, tmp_path):
        db = str(tmp_path / "test.db")
        logger = AttendanceLogger(db_path=db)
        logger.log("Alice", "Happy")
        records = logger.get_all_records()
        assert len(records) == 1
        assert records[0][1] == "Alice"
        assert records[0][2] == "Happy"

    def test_stats_today(self, tmp_path):
        db = str(tmp_path / "test.db")
        logger = AttendanceLogger(db_path=db)
        logger.log("Bob",   "Neutral")
        logger.log("Carol", "Sad")
        stats = logger.stats_today()
        assert stats["total_entries"] == 2
        assert stats["unique_people"] == 2
        assert "Bob" in stats["people"]

    def test_export_csv(self, tmp_path):
        db  = str(tmp_path / "test.db")
        csv = str(tmp_path / "out.csv")
        logger = AttendanceLogger(db_path=db)
        logger.log("Dave", "Happy")
        logger.export_csv(path=csv)
        assert os.path.exists(csv)
        with open(csv) as f:
            content = f.read()
        assert "Dave" in content
        assert "Happy" in content
