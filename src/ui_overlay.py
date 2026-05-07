import cv2
import numpy as np
import datetime

EMOTION_COLORS = {
    "Happy":     (50,  220,  80),
    "Sad":       (220,  80,  50),
    "Angry":     (50,   50, 220),
    "Surprised": (50,  220, 220),
    "Excited":   (50,  255, 255),
    "Fear":      (200,  50, 220),
    "Disgust":   (70,  140,   0),
    "Neutral":   (180, 180, 180),
    "Unknown":   (128, 128, 128),
}

FONT       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
THICKNESS  = 1


def draw_overlay(frame: np.ndarray, results: list, fps: float) -> np.ndarray:
    for result in results:
        _draw_face(frame, result)
    _draw_hud(frame, fps, len(results))
    return frame


def _draw_face(frame: np.ndarray, result: dict):
    x, y, w, h = result["bbox"]
    emotion    = result.get("emotion", "Unknown")
    confidence = result.get("emotion_conf", 0.0)
    identity   = result.get("identity", "Unknown")

    color = EMOTION_COLORS.get(emotion, (200, 200, 200))

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    _draw_corners(frame, x, y, w, h, color)

    _draw_label_bg(frame, x, max(y - 28, 0), identity, color)
    cv2.putText(
        frame, identity,
        (x + 6, max(y - 6, 14)),
        FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA,
    )

    emo_text = f"{emotion}  {int(confidence * 100)}%"
    _draw_label_bg(frame, x, y + h, emo_text, color)
    cv2.putText(
        frame, emo_text,
        (x + 6, y + h + 22),
        FONT, FONT_SCALE, (255, 255, 255), THICKNESS, cv2.LINE_AA,
    )

    bar_x1   = x
    bar_x2   = x + w
    bar_y    = y + h + 26
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x2, bar_y + 4), (60, 60, 60), -1)
    filled_w = int((bar_x2 - bar_x1) * confidence)
    cv2.rectangle(frame, (bar_x1, bar_y), (bar_x1 + filled_w, bar_y + 4), color, -1)


def _draw_corners(frame, x, y, w, h, color, length=18, thickness=3):
    cv2.line(frame, (x, y),         (x + length, y),         color, thickness)
    cv2.line(frame, (x, y),         (x, y + length),         color, thickness)
    cv2.line(frame, (x + w, y),     (x + w - length, y),     color, thickness)
    cv2.line(frame, (x + w, y),     (x + w, y + length),     color, thickness)
    cv2.line(frame, (x, y + h),     (x + length, y + h),     color, thickness)
    cv2.line(frame, (x, y + h),     (x, y + h - length),     color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w - length, y + h), color, thickness)
    cv2.line(frame, (x + w, y + h), (x + w, y + h - length), color, thickness)


def _draw_label_bg(frame, x, y, text, color, padding=5):
    (tw, th), _ = cv2.getTextSize(text, FONT, FONT_SCALE, THICKNESS)
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x, y),
        (x + tw + padding * 2, y + th + padding * 2),
        color, -1,
    )
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)


def _draw_hud(frame: np.ndarray, fps: float, face_count: int):
    fh, fw = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, 70), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    now_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, f"FPS: {fps:.1f}",      (10, 22), FONT, 0.60, (100, 240, 100), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Faces: {face_count}", (10, 44), FONT, 0.60, (100, 200, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, now_str,                (10, 63), FONT, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

    bar_y    = fh - 28
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, bar_y), (fw, fh), (20, 20, 20), -1)
    cv2.addWeighted(overlay2, 0.6, frame, 0.4, 0, frame)
    cv2.putText(
        frame,
        "  [Q] Quit    [S] Screenshot    [R] Register Face",
        (10, fh - 8),
        FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
    )

    title = "Face & Emotion Detection | P2837554"
    (tw, _), _ = cv2.getTextSize(title, FONT, 0.5, 1)
    cv2.putText(
        frame, title,
        (fw - tw - 10, 22),
        FONT, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )
