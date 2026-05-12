import cv2, os, time, threading
import numpy as np
from collections import deque
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from tensorflow.keras.models import load_model
from dotenv import load_dotenv

load_dotenv()

class Config:
    W, H = 960, 540
    HOLD = 15
    PAUSE = 2.5
    GEMINI_KEY = os.getenv("GEMINI_API_KEY", "")
    ELEVEN_KEY = os.getenv("ELEVENLABS_API_KEY", "")
    VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    LABELS = {i: chr(i + 65) for i in range(26) if i != 9}

class AIEngine:
    def __init__(self):
        self.cache = {}
        self.lock = threading.Lock()
        self.last = 0.0
        self.enabled = bool(Config.GEMINI_KEY)
        if self.enabled:
            import google.generativeai as genai
            genai.configure(api_key=Config.GEMINI_KEY)
            self.model = genai.GenerativeModel("gemini-2.5-flash", 
                system_instruction="Fix ASL text. Correct spelling, add punctuation. Return ONLY the sentence.")

    def polish(self, text):
        if len(text.split()) < 3 or not self.enabled: return text.capitalize()
        if text in self.cache: return self.cache[text]
        with self.lock:
            if time.time() - self.last < 2.0: time.sleep(2.0 - (time.time() - self.last))
            self.last = time.time()
        try:
            res = self.model.generate_content(text).text.strip(' "\'')
            self.cache[text] = res 
            return res
        except: return text.capitalize()

class VoiceEngine:
    def __init__(self):
        self.enabled = bool(Config.ELEVEN_KEY)
        if self.enabled:
            from elevenlabs.client import ElevenLabs
            self.client = ElevenLabs(api_key=Config.ELEVEN_KEY)

    def speak(self, text):
        if not self.enabled or not text.strip(): return
        threading.Thread(target=self._run, args=(text,), daemon=True).start()

    def _run(self, text):
        try:
            audio = b"".join(self.client.text_to_speech.convert(
                text=text, voice_id=Config.VOICE_ID, model_id="eleven_turbo_v2_5"
            ))
            import tempfile, pygame
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                f.write(audio); tmp = f.name
            pygame.mixer.init()
            pygame.mixer.music.load(tmp)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy(): time.sleep(0.05)
            os.unlink(tmp)
        except: pass

class SystemControls:
    @staticmethod
    def check(lms):
        e = [lms[i].y < lms[i-2].y for i in [8, 12, 16, 20]]
        if all(e): return "space"
        if not any(e) and lms[4].y > lms[3].y + 0.05: return "delete"
        if e[0] and e[1] and not e[2] and not e[3]: return "peace"
        return None

def process_frame(frame, lms, cnn):
    h, w, _ = frame.shape
    x_min, y_min = w, h
    x_max, y_max = 0, 0
    for lm in lms:
        x, y = int(lm.x * w), int(lm.y * h)
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x), max(y_max, y)
    
    pad = 20
    x_min, y_min = max(0, x_min - pad), max(0, y_min - pad)
    x_max, y_max = min(w, x_max + pad), min(h, y_max + pad)
    
    if x_max - x_min < 20 or y_max - y_min < 20: return None
    
    crop = frame[y_min:y_max, x_min:x_max]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    norm = resized.reshape(1, 28, 28, 1) / 255.0
    pred = cnn.predict(norm, verbose=0)
    idx = np.argmax(pred)
    return Config.LABELS.get(idx, None)

def main():
    if not os.path.exists('model.h5'): 
        print("Model not found. Please run 'python train.py' first.")
        return
    
    # Auto-download MediaPipe task file if missing
    task_path = "hand_landmarker.task"
    if not os.path.exists(task_path):
        import urllib.request
        print("Downloading MediaPipe model, please wait...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", 
            task_path
        )

    cnn = load_model('model.h5')
    
    
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1, 
        min_hand_detection_confidence=0.7
    )
    detector = mp_vision.HandLandmarker.create_from_options(options)
    
    ai, voice = AIEngine(), VoiceEngine()
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    word, sealed = [], []
    buf = deque(maxlen=Config.HOLD)
    last_t = time.time()
    cd = 0
    ts = 0

    print("System Online. Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(cv2.resize(frame, (Config.W, Config.H)), 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process tracking
        ts += 33333
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        res = detector.detect_for_video(mp_image, ts)
        
        lbl = None
        now = time.time()

        if res.hand_landmarks:
            lms = res.hand_landmarks[0] # Grab first hand
            lbl = SystemControls.check(lms)
            if not lbl: lbl = process_frame(frame, lms, cnn)

        if not lbl and word and (now - last_t > Config.PAUSE):
            sealed.append("".join(word)); word.clear()
        
        if not lbl: 
            buf.clear()
            if cd > 0: cd -= 1
        else:
            if lbl == "space":
                if word: sealed.append("".join(word)); word.clear()
                buf.clear()
            elif lbl == "delete":
                if word: word.pop()
                elif sealed: sealed.pop()
                buf.clear()
            elif lbl == "peace":
                if word: sealed.append("".join(word)); word.clear()
                txt = " ".join(sealed).strip()
                if txt:
                    voice.speak(ai.polish(txt))
                    sealed.clear()
                buf.clear()
            else:
                buf.append(lbl)
                if len(buf) == Config.HOLD:
                    dom = max(set(buf), key=buf.count)
                    if buf.count(dom) >= Config.HOLD * 0.75 and cd == 0:
                        word.append(dom)
                        last_t = now
                        cd = 12
                        buf.clear()
            if cd > 0: cd -= 1

        cv2.putText(frame, f"Typing: {''.join(word)}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 200, 100), 2)
        cv2.putText(frame, f"Sealed: {' '.join(sealed)}", (30, 90), cv2.FONT_HERSHEY_DUPLEX, 0.8, (200, 200, 200), 1)
        cv2.imshow("Vision Clair ML", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()