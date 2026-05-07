import cv2
import numpy as np
import time
import os
import sys
import shutil
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.face_detector import FaceDetector
from src.emotion_estimator import EmotionEstimator
from src.face_recognizer import FaceRecognizer
from src.ui_overlay import draw_overlay
from src.youtube_recommender import RecommendationPanel

BG     = "#f5f5f5"
WHITE  = "#ffffff"
BLUE   = "#1a73e8"
GREEN  = "#2e7d32"
RED    = "#c62828"
TEXT   = "#212121"
MUTED  = "#757575"
BORDER = "#e0e0e0"

KNOWN_FACES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "known_faces",
)

_FACE_ROWS_HEIGHT = 140


class RegisterWindow:
    STEP_LABELS = ["1. Straight", "2. Tilt Left", "3. Tilt Right"]
    STEP_INSTRUCTIONS = [
        "Look straight at the camera.",
        "Tilt your head slightly to the LEFT.",
        "Tilt your head slightly to the RIGHT.",
    ]

    def __init__(self, parent, app):
        self.app    = app
        self.step   = 0
        self.photos = []
        self.name   = ""

        name = simpledialog.askstring(
            "Register Face", "Enter the person's name:", parent=parent
        )
        if not name or not name.strip():
            return
        self.name = name.strip()

        self.win = tk.Toplevel(parent)
        self.win.title("Register Face")
        self.win.configure(bg=WHITE)
        self.win.resizable(False, False)
        self.win.grab_set()
        self.win.protocol("WM_DELETE_WINDOW", self.cancel)
        self._build_ui()
        self._refresh()

    def _build_ui(self):
        tk.Label(
            self.win, text=f"Registering: {self.name}",
            bg=BLUE, fg=WHITE, font=("Segoe UI", 12, "bold"),
        ).pack(fill="x", ipady=10)

        step_bar = tk.Frame(self.win, bg=WHITE)
        step_bar.pack(fill="x", padx=20, pady=(16, 0))
        self.step_labels = []
        for text in self.STEP_LABELS:
            cell = tk.Frame(step_bar, bg=WHITE)
            cell.pack(side="left", expand=True, fill="x")
            lbl = tk.Label(
                cell, text=text, bg=BORDER, fg=MUTED,
                font=("Segoe UI", 9, "bold"), pady=6, relief="flat",
            )
            lbl.pack(fill="x", padx=2)
            self.step_labels.append(lbl)

        self.instruct_lbl = tk.Label(
            self.win, text="", bg=WHITE, fg=TEXT,
            font=("Segoe UI", 11), wraplength=380, justify="center",
        )
        self.instruct_lbl.pack(pady=(16, 8))

        self.preview = tk.Label(self.win, bg="black")
        self.preview.pack(padx=20, pady=(0, 12))

        self.status_lbl = tk.Label(
            self.win, text="", bg=WHITE, fg=GREEN,
            font=("Segoe UI", 9, "bold"),
        )
        self.status_lbl.pack()

        btn_row = tk.Frame(self.win, bg=WHITE)
        btn_row.pack(fill="x", padx=20, pady=16)
        tk.Button(
            btn_row, text="Capture Photo",
            bg=BLUE, fg=WHITE, font=("Segoe UI", 11, "bold"),
            relief="flat", pady=10, cursor="hand2",
            command=self.capture,
        ).pack(side="left", fill="x", expand=True, padx=(0, 8))
        tk.Button(
            btn_row, text="Cancel",
            bg=BORDER, fg=TEXT, font=("Segoe UI", 10),
            relief="flat", pady=10, command=self.cancel,
        ).pack(side="right", fill="x", expand=True)

        self.running = True
        self._update_preview()

    def _refresh(self):
        self.instruct_lbl.config(text=self.STEP_INSTRUCTIONS[self.step])
        for i, lbl in enumerate(self.step_labels):
            if i < self.step:    lbl.config(bg=GREEN,  fg=WHITE)
            elif i == self.step: lbl.config(bg=BLUE,   fg=WHITE)
            else:                lbl.config(bg=BORDER, fg=MUTED)
        self.status_lbl.config(
            text=f"Photo {self.step} of 3 captured." if self.step > 0 else ""
        )

    def _update_preview(self):
        if not self.running:
            return
        if self.app.last_frame is not None:
            frame = self.app.last_frame.copy()
            for (x, y, w, h) in self.app.last_faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (26, 115, 232), 2)
            frame = cv2.resize(frame, (380, 240))
            img   = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            imgtk = ImageTk.PhotoImage(img)
            self.preview.config(image=imgtk)
            self.preview.image = imgtk
        self.win.after(33, self._update_preview)

    def capture(self):
        if self.app.last_frame is None:
            messagebox.showwarning("Wait", "Camera not ready.", parent=self.win)
            return
        if len(self.app.last_faces) == 0:
            messagebox.showwarning(
                "No face",
                "No face detected.\nPlease position your face in the camera.",
                parent=self.win,
            )
            return
        if len(self.app.last_faces) > 1:
            messagebox.showwarning(
                "Multiple faces", "Only one face should be visible.",
                parent=self.win,
            )
            return
        self.photos.append((self.app.last_frame.copy(), self.app.last_faces[0]))
        self.step += 1
        if self.step >= 3:
            self._finish()
        else:
            self._refresh()

    def _finish(self):
        self.running = False
        for frm, bbox in self.photos:
            self.app.recognizer.register_face(frm, bbox, self.name, KNOWN_FACES_DIR)

        self.app.recognizer.known_names    = []
        self.app.recognizer.known_features = []
        self.app.recognizer.load_known_faces(KNOWN_FACES_DIR)
        self.app.set_status(f"'{self.name}' registered with 3 photos.")
        print(f"  [Registered: {self.name}]")
        self.win.destroy()
        messagebox.showinfo("Success", f"'{self.name}' registered successfully!")

    def cancel(self):
        self.running = False
        self.win.destroy()
        self.app.set_status("Registration cancelled.")


class App:

    _EMOTION_COLORS = {
        "Happy":     GREEN,     "Excited":  "#e65100",
        "Sad":       BLUE,      "Angry":    RED,
        "Surprised": "#e3e619", "Fear":     "#6a1b9a",
        "Disgust":   "#1b5e20", "Neutral":  TEXT,
    }

    def __init__(self, root):
        self.root = root
        self.root.title("Face and Emotion Detection System - P2837554")
        self.root.configure(bg=BG)
        self.root.resizable(False, False)

        self.detector    = FaceDetector()
        self.emotion_est = EmotionEstimator()
        self.recognizer  = FaceRecognizer()
        self.recognizer.load_known_faces(KNOWN_FACES_DIR)

        self.running        = True
        self.last_frame     = None
        self.last_faces     = []
        self.fps            = 0.0
        self._last_results  = []
        self._process_count = 0
        self.fps_start      = time.time()
        self.frame_count    = 0
        self.status_msg     = ""
        self._yt_visible    = False

        self._build_ui()
        self._start_camera()
        self.root.protocol("WM_DELETE_WINDOW", self.quit)

    def _build_ui(self):
        self._build_header()
        self._build_body()
        self._build_status_bar()

        self.root.bind("<s>", lambda e: self.screenshot())
        self.root.bind("<q>", lambda e: self.quit())
        self.root.bind("<r>", lambda e: self.open_register())
        self.root.bind("<S>", lambda e: self.screenshot_face())
        self.root.bind("<Q>", lambda e: self.quit())
        self.root.bind("<R>", lambda e: self.open_register())

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=BLUE, height=50)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(
            hdr, text="Face and Emotion Detection System",
            bg=BLUE, fg=WHITE, font=("Segoe UI", 13, "bold"),
        ).pack(side="left", padx=16, pady=12)
        self.clock_lbl = tk.Label(
            hdr, text="", bg=BLUE, fg="#cfe2ff", font=("Segoe UI", 9),
        )
        self.clock_lbl.pack(side="right", padx=16)

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", padx=10, pady=10)

        cam_wrap = tk.Frame(body, bg=BORDER, bd=1, relief="solid")
        cam_wrap.pack(side="left")
        self.canvas = tk.Canvas(
            cam_wrap, width=820, height=540,
            bg="black", highlightthickness=0,
        )
        self.canvas.pack()

        self._yt_container = tk.Frame(body, bg=BG, width=280)
        self._yt_container.pack(side="right", fill="y", padx=(6, 0))
        self._yt_container.pack_propagate(False)
        self.yt_panel = RecommendationPanel(self._yt_container)

        stats_frame = tk.Frame(body, bg=BG, width=240)
        stats_frame.pack(side="right", fill="y", padx=(10, 6))
        stats_frame.pack_propagate(False)
        self._build_stats_panel(stats_frame)

    def _build_status_bar(self):
        tk.Frame(self.root, bg=BORDER, height=1).pack(fill="x")
        self.status_bar = tk.Label(
            self.root, text="Ready.",
            bg=WHITE, fg=MUTED, font=("Segoe UI", 9), anchor="w",
        )
        self.status_bar.pack(fill="x", ipady=5, padx=10)

    def _section_label(self, parent, text):
        tk.Label(
            parent, text=text,
            bg=BG, fg=MUTED, font=("Segoe UI", 8, "bold"),
        ).pack(anchor="w", pady=(10, 2))

    def _build_stats_panel(self, parent):
        self._section_label(parent, "LIVE STATS")
        c = tk.Frame(parent, bg=WHITE, relief="solid", bd=1)
        c.pack(fill="x")
        self.fps_lbl   = tk.Label(c, text="FPS: --",   bg=WHITE, fg=TEXT, font=("Segoe UI", 9), anchor="w")
        self.faces_lbl = tk.Label(c, text="Faces: --", bg=WHITE, fg=TEXT, font=("Segoe UI", 9), anchor="w")
        self.known_lbl = tk.Label(c, text="Known: --", bg=WHITE, fg=TEXT, font=("Segoe UI", 9), anchor="w")
        for lbl in (self.fps_lbl, self.faces_lbl, self.known_lbl):
            lbl.pack(fill="x", padx=8, pady=1)

        self._section_label(parent, "DETECTED FACES")
        self._face_rows_frame = tk.Frame(
            parent, bg=BG,
            height=_FACE_ROWS_HEIGHT,
        )
        self._face_rows_frame.pack(fill="x")
        self._face_rows_frame.pack_propagate(False)

        self._section_label(parent, "REGISTER FACE")
        tk.Button(
            parent, text="Register New Face",
            bg=GREEN, fg=WHITE, font=("Segoe UI", 9, "bold"),
            relief="flat", pady=7, cursor="hand2",
            command=self.open_register,
        ).pack(fill="x")

        self._section_label(parent, "MANAGE")
        tk.Button(
            parent, text="Delete a Registered Face",
            bg=RED, fg=WHITE, font=("Segoe UI", 9, "bold"),
            relief="flat", pady=7, cursor="hand2",
            command=self.delete_face,
        ).pack(fill="x")
        self.names_lbl = tk.Label(
            parent, text="", bg=BG, fg=MUTED,
            font=("Segoe UI", 8), justify="left", wraplength=220,
        )
        self.names_lbl.pack(anchor="w", pady=(3, 0))

        self._section_label(parent, "ACTIONS")
        tk.Button(
            parent, text="Save Screenshot",
            bg=BLUE, fg=WHITE, font=("Segoe UI", 9, "bold"),
            relief="flat", pady=7, cursor="hand2",
            command=self.screenshot,
        ).pack(fill="x", pady=(0, 4))
        tk.Button(
            parent, text="Quit",
            bg=RED, fg=WHITE, font=("Segoe UI", 9, "bold"),
            relief="flat", pady=7, cursor="hand2",
            command=self.quit,
        ).pack(fill="x")

    def _start_camera(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self._update_frame()

    def _update_frame(self):
        if not self.running:
            return

        ret, frame = self.cap.read()
        if not ret:
            self.root.after(30, self._update_frame)
            return

        frame = cv2.flip(frame, 1)

        self.frame_count += 1
        elapsed = time.time() - self.fps_start
        if elapsed >= 1.0:
            self.fps         = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start   = time.time()

        self.last_frame      = frame.copy()
        self._process_count += 1

        if self._process_count % 2 == 0:
            faces   = self.detector.detect(frame)
            results = []
            for (x, y, w, h) in faces:
                face_roi      = frame[y:y + h, x:x + w]
                emotion, conf = self.emotion_est.estimate(frame, (x, y, w, h))
                identity      = self.recognizer.recognize_stable(face_roi)
                results.append({
                    "bbox":         (x, y, w, h),
                    "emotion":      emotion,
                    "emotion_conf": conf,
                    "identity":     identity,
                })
            self.last_faces    = faces
            self._last_results = results

        display = draw_overlay(frame.copy(), self._last_results, self.fps)
        display = cv2.resize(display, (820, 540))
        img     = Image.fromarray(cv2.cvtColor(display, cv2.COLOR_BGR2RGB))
        imgtk   = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor="nw", image=imgtk)
        self.canvas.image = imgtk

        self._update_stats(self._last_results)
        self.root.after(16, self._update_frame)

    def _update_stats(self, results):
        face_detected = len(results) > 0

        self.clock_lbl.config(
            text=datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
        )
        self.fps_lbl.config(text=f"FPS:   {self.fps:.1f}")
        self.faces_lbl.config(text=f"Faces: {len(results)}")
        self.known_lbl.config(text=f"Known: {len(set(self.recognizer.known_names))}")

        self._update_face_rows(results)

        if face_detected:
            if not self._yt_visible:
                self.yt_panel.show()
                self._yt_visible = True
            self.yt_panel.update_emotion(results[0].get("emotion", "Neutral"))
        else:
            if self._yt_visible:
                self.yt_panel.hide()
                self._yt_visible = False

        self.status_bar.config(
            text=self.status_msg if self.status_msg else "Ready."
        )

        names = sorted(set(self.recognizer.known_names))
        self.names_lbl.config(
            text="Registered: " + ", ".join(names) if names else "No registered faces."
        )

    def _update_face_rows(self, results):
        for w in self._face_rows_frame.winfo_children():
            w.destroy()

        if not results:
            tk.Label(
                self._face_rows_frame,
                text="No face detected.",
                bg=BG, fg=MUTED, font=("Segoe UI", 8),
            ).pack(anchor="w", pady=4, padx=4)
            return

        for idx, r in enumerate(results[:4]):
            emotion  = r.get("emotion",      "--")
            conf     = r.get("emotion_conf", 0.0)
            identity = r.get("identity",     "Unknown")
            color    = self._EMOTION_COLORS.get(emotion, TEXT)

            card = tk.Frame(
                self._face_rows_frame,
                bg=WHITE, relief="solid", bd=1,
            )
            card.pack(fill="x", pady=(0, 3))

            tk.Label(
                card, text=f"#{idx + 1}",
                bg=color, fg=WHITE,
                font=("Segoe UI", 8, "bold"),
                width=3, pady=3,
            ).pack(side="left")

            info = tk.Frame(card, bg=WHITE)
            info.pack(side="left", fill="x", expand=True, padx=5)

            tk.Label(
                info, text=identity,
                bg=WHITE, fg=TEXT,
                font=("Segoe UI", 8, "bold"), anchor="w",
            ).pack(fill="x")

            row2 = tk.Frame(info, bg=WHITE)
            row2.pack(fill="x")
            tk.Label(
                row2, text=emotion,
                bg=WHITE, fg=color, font=("Segoe UI", 8, "bold"),
            ).pack(side="left")
            tk.Label(
                row2, text=f"  {int(conf * 100)}%",
                bg=WHITE, fg=MUTED, font=("Segoe UI", 8),
            ).pack(side="left")

    def open_register(self):
        RegisterWindow(self.root, self)

    def delete_face(self):
        names = sorted(set(self.recognizer.known_names))
        if not names:
            messagebox.showinfo("No faces", "No registered faces to delete.")
            return

        win = tk.Toplevel(self.root)
        win.title("Delete Registered Face")
        win.configure(bg=WHITE)
        win.resizable(False, False)
        win.grab_set()

        tk.Label(
            win, text="Select a person to delete:",
            bg=WHITE, fg=TEXT, font=("Segoe UI", 10, "bold"),
        ).pack(padx=24, pady=(18, 8))

        var = tk.StringVar(value=names[0])
        for name in names:
            tk.Radiobutton(
                win, text=name, variable=var, value=name,
                bg=WHITE, fg=TEXT, font=("Segoe UI", 10),
                activebackground=WHITE,
            ).pack(anchor="w", padx=28, pady=3)

        tk.Frame(win, bg=BORDER, height=1).pack(fill="x", pady=10)

        def confirm():
            selected = var.get()
            if not messagebox.askyesno(
                "Confirm", f"Delete all data for '{selected}'?", parent=win
            ):
                return
            person_dir = os.path.join(KNOWN_FACES_DIR, selected)
            if os.path.isdir(person_dir):
                shutil.rmtree(person_dir)
            pkl = os.path.join(KNOWN_FACES_DIR, "features.pkl")
            if os.path.exists(pkl):
                os.remove(pkl)
            self.recognizer.known_names    = []
            self.recognizer.known_features = []
            self.recognizer.load_known_faces(KNOWN_FACES_DIR)
            self.set_status(f"Deleted '{selected}'.")
            print(f"  [Deleted: {selected}]")
            win.destroy()

        btn_f = tk.Frame(win, bg=WHITE)
        btn_f.pack(fill="x", padx=24, pady=(0, 18))
        tk.Button(
            btn_f, text="Delete",
            bg=RED, fg=WHITE, font=("Segoe UI", 9, "bold"),
            relief="flat", pady=6, command=confirm,
        ).pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Button(
            btn_f, text="Cancel",
            bg=BORDER, fg=TEXT, font=("Segoe UI", 9),
            relief="flat", pady=6, command=win.destroy,
        ).pack(side="right", fill="x", expand=True)

    def screenshot(self):
        if self.last_frame is None:
            return
        path = f"screenshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(path, self.last_frame)
        self.set_status(f"Screenshot saved: {path}")
        print(f"  [Screenshot: {path}]")

    def screenshot_face(self):
        if self.last_frame is None or not self.last_faces:
            return
        x, y, w, h = self.last_faces[0]
        face = self.last_frame[y:y+h, x:x+w]
        path = f"face_{time.strftime('%Y%m%d_%H%M%S')}.png"
        cv2.imwrite(path, face)
        self.set_status(f"Face screenshot saved: {path}")
        print(f"  [Face Screenshot: {path}]")

    def set_status(self, msg):
        self.status_msg = msg

    def quit(self):
        self.running = False
        if hasattr(self, "cap"):
            self.cap.release()
        self.root.destroy()


def main():
    print("=" * 55)
    print("  Face and Emotion Detection System")
    print("  P2837554 | Ritika Thapa Magar")
    print("=" * 55)

    root  = tk.Tk()
    style = ttk.Style()
    style.theme_use("default")
    style.configure(
        "Horizontal.TProgressbar",
        troughcolor=BORDER, background=BLUE, thickness=12,
    )
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
