import customtkinter as ctk
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import mss
import threading
import time
import os
import tkinter as tk
from tkinter import filedialog


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_FILE = os.path.join(BASE_DIR, "models", "best_model_pytorch_finetuned.pth")

# --- CONFIGURATION ---
MODEL_NAME = "resnet50"
NUM_CLASSES = 1
IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD = 0.5

# --- THEME SETUP ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

# --- AI ENGINE ---
class AIEngine:
    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"AI Engine initialized on: {self.device}")
        self.model = self.load_model()
        self.face_cascade = self.load_cascade()
        self.transforms = transforms.Compose([
            transforms.Resize(IMAGE_SIZE + 32),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self):
        try:
            if not os.path.exists(MODEL_FILE):
                print(f"CRITICAL ERROR: Model file not found at {MODEL_FILE}")
                return None
            
            model = models.resnet50()
            model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
            # Use weights_only=True for security
            model.load_state_dict(torch.load(MODEL_FILE, map_location=self.device, weights_only=True))
            model.eval()
            model.to(self.device)
            print("Model loaded successfully.")
            return model
        except Exception as e:
            print(f"Model Load Error: {e}")
            return None

    def load_cascade(self):
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        return cv2.CascadeClassifier(path)

    def preprocess(self, face_img):
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
        return self.transforms(pil_img).unsqueeze(0).to(self.device)

    @torch.no_grad()
    def predict(self, face_img):
        if self.model is None: return 0.5
        tensor = self.preprocess(face_img)
        output = self.model(tensor)
        return torch.sigmoid(output).item()

# --- HUD OVERLAY SYSTEM ---
class ScreenOverlay:
    def __init__(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.windows = []
        self.color = "green"
        self.thickness = 0
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.is_visible = False
        
        self.create_border_window("top")
        self.create_border_window("bottom")
        self.create_border_window("left")
        self.create_border_window("right")

    def create_border_window(self, side):
        win = tk.Toplevel(self.root)
        win.overrideredirect(True)
        win.attributes("-topmost", True)
        win.attributes("-alpha", 0.0)
        win.config(bg="green")
        self.windows.append((win, side))

    def update_overlay(self, status):
        if status == 'idle':
            if self.is_visible:
                for win, _ in self.windows: win.attributes("-alpha", 0.0)
                self.is_visible = False
            return

        self.is_visible = True
        if status == 'danger':
            color = "#ff0000"
            thickness = 10
        else:
            color = "#00ff00"
            thickness = 5

        for win, side in self.windows:
            win.config(bg=color)
            win.attributes("-alpha", 0.8)
            
            if side == "top":
                win.geometry(f"{self.screen_width}x{thickness}+0+0")
            elif side == "bottom":
                win.geometry(f"{self.screen_width}x{thickness}+0+{self.screen_height-thickness}")
            elif side == "left":
                win.geometry(f"{thickness}x{self.screen_height}+0+0")
            elif side == "right":
                win.geometry(f"{thickness}x{self.screen_height}+{self.screen_width-thickness}+0")
        
        self.root.update()

# --- MAIN APPLICATION GUI ---
class SentinelApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Deepfake Sentinel Command")
        self.geometry("900x650")
        
        self.ai = AIEngine()
        self.overlay = ScreenOverlay()
        self.guardian_running = False
        self.webcam_running = False
        self.cap = None
        
        self.create_sidebar()
        self.create_main_area()
        self.show_file_analyzer()

    def create_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        sidebar.pack(side="left", fill="y")
        
        logo = ctk.CTkLabel(sidebar, text="SENTINEL\nAI DEFENSE", font=("Roboto Medium", 20, "bold"))
        logo.pack(pady=30)
        
        btn_file = ctk.CTkButton(sidebar, text="File Scanner", command=self.show_file_analyzer, fg_color="transparent", border_width=2)
        btn_file.pack(pady=10, padx=20, fill="x")
        
        btn_cam = ctk.CTkButton(sidebar, text="Webcam Mirror", command=self.show_webcam_mirror, fg_color="transparent", border_width=2)
        btn_cam.pack(pady=10, padx=20, fill="x")
        
        btn_guard = ctk.CTkButton(sidebar, text="Screen Guardian", command=self.show_guardian, fg_color="transparent", border_width=2)
        btn_guard.pack(pady=10, padx=20, fill="x")
        
        self.status_lbl = ctk.CTkLabel(sidebar, text="Status: SYSTEM READY", text_color="green", font=("Consolas", 12))
        self.status_lbl.pack(side="bottom", pady=20)

    def create_main_area(self):
        self.main_frame = ctk.CTkFrame(self, corner_radius=10)
        self.main_frame.pack(side="right", fill="both", expand=True, padx=20, pady=20)

    def clear_frame(self):
        for widget in self.main_frame.winfo_children():
            widget.destroy()

    def show_file_analyzer(self):
        self.clear_frame()
        title = ctk.CTkLabel(self.main_frame, text="Forensic File Scanner", font=("Roboto Medium", 24))
        title.pack(pady=20)
        self.file_path_lbl = ctk.CTkLabel(self.main_frame, text="No file selected", text_color="gray")
        self.file_path_lbl.pack(pady=5)
        btn_browse = ctk.CTkButton(self.main_frame, text="Select Media", command=self.browse_file)
        btn_browse.pack(pady=10)
        self.file_result_frame = ctk.CTkFrame(self.main_frame, fg_color="#1a1a1a", height=100)
        self.file_result_frame.pack(fill="x", padx=40, pady=20)
        self.file_res_text = ctk.CTkLabel(self.file_result_frame, text="WAITING FOR INPUT", font=("Consolas", 28, "bold"), text_color="gray")
        self.file_res_text.place(relx=0.5, rely=0.5, anchor="center")

    def browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("Media", "*.jpg *.png *.jpeg *.mp4 *.avi *.mov")])
        if path:
            self.file_path_lbl.configure(text=os.path.basename(path))
            threading.Thread(target=self.analyze_file_thread, args=(path,), daemon=True).start()

    def analyze_file_thread(self, path):
        self.file_res_text.configure(text="ANALYZING...", text_color="yellow")
        if path.lower().endswith(('.jpg', '.png', '.jpeg')):
            frame = cv2.imread(path)
            res_text, color = self.process_single_frame(frame)
        else:
            cap = cv2.VideoCapture(path)
            fakes, total = 0, 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if total % 20 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.ai.face_cascade.detectMultiScale(gray, 1.1, 4)
                    for (x, y, w, h) in faces:
                        roi = frame[y:y+h, x:x+w]
                        if roi.size > 0:
                            score = self.ai.predict(roi)
                            if score < 0.5: fakes += 1
                            total += 1
            cap.release()
            if total == 0: res_text, color = "NO FACES FOUND", "gray"
            else:
                ratio = fakes / total
                res_text, color = (f"FAKE DETECTED ({ratio*100:.1f}%)", "#ff3333") if ratio > 0.3 else ("AUTHENTIC MEDIA", "#33ff33")
        self.file_res_text.configure(text=res_text, text_color=color)

    def process_single_frame(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.ai.face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) == 0: return "NO FACE DETECTED", "gray"
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        (x, y, w, h) = faces[0]
        score = self.ai.predict(frame[y:y+h, x:x+w])
        return (f"REAL ({score*100:.1f}%)", "#33ff33") if score > CONFIDENCE_THRESHOLD else (f"FAKE ({(1-score)*100:.1f}%)", "#ff3333")

    def show_webcam_mirror(self):
        self.stop_guardian_mode()
        self.clear_frame()
        ctk.CTkLabel(self.main_frame, text="Webcam Security Mirror", font=("Roboto Medium", 24)).pack(pady=10)
        self.cam_view = ctk.CTkLabel(self.main_frame, text="")
        self.cam_view.pack(pady=10)
        ctk.CTkButton(self.main_frame, text="Toggle Camera", command=self.toggle_webcam).pack(pady=10)

    def toggle_webcam(self):
        if self.webcam_running:
            self.webcam_running = False
            if self.cap: self.cap.release()
        else:
            self.webcam_running = True
            self.cap = cv2.VideoCapture(0)
            self.update_webcam()

    def update_webcam(self):
        if not self.webcam_running: return
        ret, frame = self.cap.read()
        if ret:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.ai.face_cascade.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    score = self.ai.predict(roi)
                    color = (0, 255, 0) if score > CONFIDENCE_THRESHOLD else (0, 0, 255)
                    label = "REAL" if score > CONFIDENCE_THRESHOLD else "FAKE"
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(frame, f"{label} {int(score*100 if label=='REAL' else (1-score)*100)}%", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.cam_view.configure(image=ctk.CTkImage(Image.fromarray(img), size=(640, 480)))
        self.after(30, self.update_webcam)

    def show_guardian(self):
        self.stop_webcam_mode()
        self.clear_frame()
        ctk.CTkLabel(self.main_frame, text="Sentinel Screen Guardian", font=("Roboto Medium", 24)).pack(pady=20)
        ctk.CTkLabel(self.main_frame, text="Monitors screen for deepfakes.", font=("Roboto", 14)).pack(pady=20)
        self.guardian_status = ctk.CTkLabel(self.main_frame, text="GUARDIAN IS OFFLINE", font=("Consolas", 20, "bold"), text_color="gray")
        self.guardian_status.pack(pady=40)
        self.btn_guardian = ctk.CTkButton(self.main_frame, text="ACTIVATE GUARDIAN", command=self.toggle_guardian, height=50, fg_color="green")
        self.btn_guardian.pack(pady=20)

    def toggle_guardian(self):
        if self.guardian_running: self.stop_guardian_mode()
        else:
            self.guardian_running = True
            self.guardian_status.configure(text="GUARDIAN IS SCANNING...", text_color="#00ff00")
            self.btn_guardian.configure(text="DEACTIVATE", fg_color="red")
            threading.Thread(target=self.guardian_loop, daemon=True).start()

    def stop_guardian_mode(self):
        self.guardian_running = False
        self.overlay.update_overlay('idle')
        if hasattr(self, 'guardian_status'): self.guardian_status.configure(text="GUARDIAN IS OFFLINE", text_color="gray")
        if hasattr(self, 'btn_guardian'): self.btn_guardian.configure(text="ACTIVATE GUARDIAN", fg_color="green")

    def stop_webcam_mode(self):
        self.webcam_running = False
        if self.cap: self.cap.release()

    def guardian_loop(self):
        sct = mss.mss()
        monitor = sct.monitors[1]
        while self.guardian_running:
            frame = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2BGR)
            small = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            faces = self.ai.face_cascade.detectMultiScale(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), 1.1, 5)
            status = 'idle'
            if len(faces) > 0:
                faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
                (x, y, w, h) = [v*2 for v in faces[0]]
                roi = frame[y:y+h, x:x+w]
                if roi.size > 0:
                    status = 'danger' if self.ai.predict(roi) < CONFIDENCE_THRESHOLD else 'safe'
            self.after(0, lambda s=status: self.overlay.update_overlay(s))
            time.sleep(0.1)

    def on_closing(self):
        self.stop_webcam_mode(); self.stop_guardian_mode(); self.destroy(); os._exit(0)

if __name__ == "__main__":
    app = SentinelApp()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()