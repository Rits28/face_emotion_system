import os
import tkinter as tk
from dataclasses import dataclass


# Data Model

@dataclass
class VideoCard:
    title:   str
    channel: str
    url:     str
    emoji:   str = "▶"


# Emotion-Based Recommendations

RECOMMENDATIONS = {
    "Happy": [
        VideoCard("Lofi Beats – Happy & Chill",   "Lofi Girl",           "https://www.youtube.com/watch?v=jfKfPfyJRdk", "😄"),
        VideoCard("Morning Relaxing Music",         "Cat Trumpet",         "https://www.youtube.com/watch?v=1ZYbU82GVz4", "✨"),
        VideoCard("Feel‑Good Instrumental Mix",     "Soothing Relaxation", "https://www.youtube.com/watch?v=2OEL4P1Rz04", "🌈"),
        VideoCard("Calm & Uplifting Piano",         "Relaxing White Noise","https://www.youtube.com/watch?v=1ZYbU82GVz4", "🌟"),
    ],
    "Sad": [
        VideoCard("Soft Lofi for Comfort",          "Lofi Girl",           "https://www.youtube.com/watch?v=kgx4WGK0oNU", "📚"),
        VideoCard("Emotional Piano Music",           "Cat Trumpet",         "https://www.youtube.com/watch?v=1ZYbU82GVz4", "🎵"),
        VideoCard("Rain Sounds + Soft Music",        "Relaxing White Noise","https://www.youtube.com/watch?v=OdIJ2x3nxzQ", "🌧"),
        VideoCard("Calming Ambient Music",           "Soothing Relaxation", "https://www.youtube.com/watch?v=2OEL4P1Rz04", "🌿"),
    ],
    "Angry": [
        VideoCard("10‑Minute Meditation Music",     "Great Meditation",         "https://www.youtube.com/watch?v=O-6f5wQXSu8", "🧘"),
        VideoCard("Ocean Waves + Calm Music",        "Nature Relaxation Films",  "https://www.youtube.com/watch?v=V1LhC1zGouc", "🌊"),
        VideoCard("Deep Relaxation Piano",           "Soothing Relaxation",      "https://www.youtube.com/watch?v=1ZYbU82GVz4", "🎹"),
        VideoCard("Forest Ambience + Music",         "Relaxing White Noise",     "https://www.youtube.com/watch?v=OdIJ2x3nxzQ", "🍃"),
    ],
    "Surprised": [
        VideoCard("Ethereal Ambient Music",          "Ambient World",            "https://www.youtube.com/watch?v=2OEL4P1Rz04", "🤯"),
        VideoCard("Dreamy Lofi Mix",                 "Lofi Girl",                "https://www.youtube.com/watch?v=jfKfPfyJRdk", "🔮"),
        VideoCard("Calm Synthwave Ambience",         "Relaxing Synthwave",       "https://www.youtube.com/watch?v=1ZYbU82GVz4", "✨"),
        VideoCard("Mystical Ambient Pads",           "Meditation Relax Music",   "https://www.youtube.com/watch?v=2OEL4P1Rz04", "🌌"),
    ],
    "Fear": [
        VideoCard("Anxiety Relief Music",            "Great Meditation",         "https://www.youtube.com/watch?v=O-6f5wQXSu8", "🧘"),
        VideoCard("Soft Piano + Rain",               "Relaxing White Noise",     "https://www.youtube.com/watch?v=OdIJ2x3nxzQ", "🌧"),
        VideoCard("Warm Lofi for Comfort",           "Lofi Girl",                "https://www.youtube.com/watch?v=kgx4WGK0oNU", "🎶"),
        VideoCard("Deep Sleep & Calm Music",         "Soothing Relaxation",      "https://www.youtube.com/watch?v=2OEL4P1Rz04", "🌙"),
    ],
    "Neutral": [
        VideoCard("Lofi Beats to Study",             "Lofi Girl",                "https://www.youtube.com/watch?v=jfKfPfyJRdk", "📚"),
        VideoCard("Peaceful Piano Music",            "Cat Trumpet",              "https://www.youtube.com/watch?v=1ZYbU82GVz4", "🎹"),
        VideoCard("Ambient Focus Music",             "Ambient World",            "https://www.youtube.com/watch?v=2OEL4P1Rz04", "🌿"),
        VideoCard("Nature Sounds + Soft Pads",       "Relaxing White Noise",     "https://www.youtube.com/watch?v=OdIJ2x3nxzQ", "🌍"),
    ],
}


# UI Constants

WHITE = "#ffffff"
BG    = "#f5f5f5"
BLUE  = "#1a73e8"
TEXT  = "#212121"
MUTED = "#757575"

EMOTION_ACCENT = {
    "Happy":     "#2e7d32",
    "Sad":       "#1565c0",
    "Angry":     "#b71c1c",
    "Surprised": "#00838f",
    "Excited":   "#e65100",
    "Fear":      "#6a1b9a",
    "Disgust":   "#1b5e20",
    "Neutral":   "#455a64",
}

EMOTION_EMOJI = {
    "Happy":     "😊",
    "Sad":       "😢",
    "Angry":     "😠",
    "Surprised": "😲",
    "Excited":   "🤩",
    "Fear":      "😨",
    "Disgust":   "🤢",
    "Neutral":   "😐",
}

DEBOUNCE_FRAMES = 40


# Recommendation Panel

class RecommendationPanel:

    def __init__(self, parent):
        self.recommendations   = RECOMMENDATIONS
        self._current_emotion  = None
        self._pending_emotion  = None
        self._pending_count    = 0

        self.frame = tk.Frame(parent, bg=BG)
        self._build_skeleton()

    def show(self):
        self.frame.pack(fill="both", expand=True)

    def hide(self):
        self.frame.pack_forget()
        self._current_emotion = None
        self._pending_emotion = None
        self._pending_count   = 0

    def update_emotion(self, emotion: str):
        if emotion == self._current_emotion:
            return

        if emotion == self._pending_emotion:
            self._pending_count += 1
        else:
            self._pending_emotion = emotion
            self._pending_count   = 1

        if self._pending_count >= DEBOUNCE_FRAMES:
            self._current_emotion = emotion
            self._pending_emotion = None
            self._pending_count   = 0
            self._render(emotion)

    # Internal builders

    def _build_skeleton(self):
        self._header = tk.Frame(self.frame, bg=MUTED, height=36)
        self._header.pack(fill="x")
        self._header.pack_propagate(False)

        self._header_lbl = tk.Label(
            self._header, text="🎬  Recommended for you",
            bg=MUTED, fg=WHITE, font=("Segoe UI", 9, "bold"),
        )
        self._header_lbl.pack(side="left", padx=10, pady=6)

        self._badge_lbl = tk.Label(
            self._header, text="",
            bg=MUTED, fg=WHITE, font=("Segoe UI", 9),
        )
        self._badge_lbl.pack(side="right", padx=10, pady=6)

        self._card_frame = tk.Frame(self.frame, bg=BG)
        self._card_frame.pack(fill="both", expand=True, padx=6, pady=6)

        tk.Label(
            self._card_frame, text="Waiting for emotion…",
            bg=BG, fg=MUTED, font=("Segoe UI", 9),
        ).pack(pady=24)

    def _render(self, emotion: str):
        for w in self._card_frame.winfo_children():
            w.destroy()

        accent = EMOTION_ACCENT.get(emotion, MUTED)
        emoji  = EMOTION_EMOJI.get(emotion, "")
        videos = self.recommendations.get(emotion, self.recommendations["Neutral"])

        self._header.config(bg=accent)
        self._header_lbl.config(bg=accent)
        self._badge_lbl.config(bg=accent, text=f"{emoji} {emotion}")

        for i, video in enumerate(videos):
            self._build_card(video, accent, i)

    def _build_card(self, video: VideoCard, accent: str, index: int):
        card_bg = WHITE if index % 2 == 0 else "#f4f4f4"

        card = tk.Frame(
            self._card_frame, bg=card_bg,
            relief="solid", bd=1, cursor="hand2",
        )
        card.pack(fill="x", pady=(0, 5))

        tk.Frame(card, bg=accent, width=5).pack(side="left", fill="y")

        content = tk.Frame(card, bg=card_bg)
        content.pack(side="left", fill="x", expand=True, padx=8, pady=5)

        title_row = tk.Frame(content, bg=card_bg)
        title_row.pack(fill="x")

        tk.Label(
            title_row, text=video.emoji,
            bg=card_bg, fg=accent, font=("Segoe UI", 11),
        ).pack(side="left")

        tk.Label(
            title_row, text=video.title,
            bg=card_bg, fg=TEXT,
            font=("Segoe UI", 8, "bold"),
            wraplength=210, justify="left", anchor="w",
        ).pack(side="left", padx=(4, 0), fill="x", expand=True)

        tk.Label(
            content, text=video.channel,
            bg=card_bg, fg=MUTED, font=("Segoe UI", 7), anchor="w",
        ).pack(fill="x")

        link = tk.Label(
            content, text="▶  Watch on YouTube",
            bg=card_bg, fg=BLUE,
            font=("Segoe UI", 8, "underline"),
            anchor="w", cursor="hand2",
        )
        link.pack(fill="x")

        def bind_events(widget, url=video.url, c=card, base_bg=card_bg):
            widget.bind("<Button-1>", lambda e: os.startfile(url))
            widget.bind("<Enter>",    lambda e: c.config(bg="#ddeeff"))
            widget.bind("<Leave>",    lambda e: c.config(bg=base_bg))

        for w in (card, content, title_row, link):
            bind_events(w)