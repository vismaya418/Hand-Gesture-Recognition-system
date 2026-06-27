import os
import csv
import logging
from datetime import datetime
import numpy as np
import queue
import threading
import pyttsx3
from src import config

# ==========================================
# Logging Configuration
# ==========================================

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(config.APPLICATION_LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HandGestureSystem")

def log_action_to_csv(gesture: str, confidence: float, action: str, duration: float):
    """
    Logs gesture and system action details to a CSV file.
    """
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    file_exists = os.path.exists(config.GESTURE_LOG_PATH)
    
    try:
        with open(config.GESTURE_LOG_PATH, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["Date", "Time", "Gesture", "Confidence", "Action", "Duration"])
            writer.writerow([date_str, time_str, gesture, f"{confidence:.2f}", action, f"{duration:.3f}"])
        logger.info(f"Logged Action: {action} (Gesture: {gesture}, Conf: {confidence:.2f})")
    except Exception as e:
        logger.error(f"Failed to log action to CSV: {e}")

# ==========================================
# Preprocessing Landmarks
# ==========================================

def preprocess_landmarks(landmarks) -> list:
    """
    Preprocess list of MediaPipe Hand Landmarks.
    1. Extracts X and Y coordinates.
    2. Subtracts wrist coordinate (index 0) to make translation-invariant.
    3. Normalizes all coordinates by maximum distance from wrist (scale-invariant).
    4. Flattens into a 42-dimensional 1D list.
    """
    # Extract coordinates
    coords = []
    for lm in landmarks:
        if hasattr(lm, 'x'):
            coords.append([lm.x, lm.y])
        else:
            coords.append([lm[0], lm[1]])
            
    coords = np.array(coords)  # Shape (21, 2)
    
    # 1. Shift origin to wrist (landmark 0)
    origin = coords[0]
    shifted = coords - origin
    
    # 2. Find max distance from origin for scaling
    distances = np.linalg.norm(shifted, axis=1)
    max_dist = np.max(distances)
    
    # 3. Scale normalize
    if max_dist > 0:
        normalized = shifted / max_dist
    else:
        normalized = shifted
        
    # 4. Flatten to a 1D list of 42 features
    return normalized.flatten().tolist()

# ==========================================
# Async TTS Voice Assistant
# ==========================================

class VoiceAssistant(threading.Thread):
    """
    Asynchronous Text-to-Speech Engine using a Queue and a background thread.
    Prevents OpenCV frames drop and UI lag from synchronous pyttsx3 calls.
    """
    def __init__(self):
        super().__init__()
        self.queue = queue.Queue()
        self.daemon = True
        self.running = True
        self.engine = None
        self.start()

    def speak(self, text: str):
        """Queue text to be spoken by the background worker."""
        if text:
            self.queue.put(text)

    def run(self):
        # Initialize the TTS engine in this background thread (required for COM on Windows)
        try:
            self.engine = pyttsx3.init()
            # Clean pronunciation settings
            self.engine.setProperty("rate", 140)
            self.engine.setProperty("volume", 0.9)
        except Exception as e:
            logger.error(f"Failed to initialize pyttsx3 in worker thread: {e}")
            self.running = False
            return

        while self.running:
            try:
                # Wait for text to speak (timeout to check running state)
                text = self.queue.get(timeout=0.5)
                logger.info(f"TTS speaking: '{text}'")
                self.engine.say(text)
                self.engine.runAndWait()
                self.queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in TTS execution thread: {e}")

    def stop(self):
        self.running = False
        # Inject dummy item to break blocked get if needed
        self.queue.put(None)
