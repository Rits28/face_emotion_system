"""
face_detector.py - Fast face detection using threading to prevent lag.
"""
import cv2
import numpy as np
import threading

try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False


class FaceDetector:
    def __init__(self):
        if DLIB_AVAILABLE:
            self.detector = dlib.get_frontal_face_detector()
            self._mode = "dlib"
        else:
            self._mode = "haar"

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Failed to load Haar Cascade from: {cascade_path}")

        self._last_faces  = []
        self._lock        = threading.Lock()
        self._processing  = False

    def detect(self, frame: np.ndarray) -> list:
        """
        Non-blocking detection — runs in background thread.
        Returns last known result immediately so camera never waits.
        """
        if not self._processing:
            self._processing = True
            t = threading.Thread(target=self._detect_thread,
                                 args=(frame.copy(),), daemon=True)
            t.start()

        with self._lock:
            return list(self._last_faces)

    def _detect_thread(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Downscale for speed
        small = cv2.resize(gray, (0, 0), fx=0.5, fy=0.5)

        if self._mode == "dlib":
            dets  = self.detector(small, 0)
            faces = [(d.left()*2, d.top()*2, d.width()*2, d.height()*2)
                     for d in dets]
        else:
            detected = self.cascade.detectMultiScale(
                small, scaleFactor=1.1, minNeighbors=4,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            faces = [] if len(detected) == 0 else [
                (int(x*2), int(y*2), int(w*2), int(h*2))
                for (x, y, w, h) in detected
            ]

        with self._lock:
            self._last_faces = faces
        self._processing = False