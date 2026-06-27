import os
from pathlib import Path

# Base Paths (resolve absolutely relative to workspace root)
ROOT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = ROOT_DIR / "dataset"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
SCREENSHOTS_DIR = ROOT_DIR / "screenshots"
ASSETS_DIR = ROOT_DIR / "assets"

# Ensure directories exist
for directory in [DATASET_DIR, MODELS_DIR, LOGS_DIR, SCREENSHOTS_DIR, ASSETS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Dataset Config
DATASET_PATH = DATASET_DIR / "gesture_data.csv"

# Model Config
MODEL_PATH = MODELS_DIR / "gesture_model.h5"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.pkl"

# Log Config
APPLICATION_LOG_PATH = LOGS_DIR / "application.log"
GESTURE_LOG_PATH = LOGS_DIR / "gesture_log.csv"

# Camera Config
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Gesture Settings
CONFIDENCE_THRESHOLD = 0.85

# Cooldown Settings (in seconds)
DISCRETE_ACTION_COOLDOWN = 2.0
CONTINUOUS_ACTION_COOLDOWN = 0.2  # For volume/brightness steps

# Gesture Labels (Canonical list)
GESTURES = [
    "Thumbs Up",
    "Thumbs Down",
    "Peace",
    "Fist",
    "Open Palm",
    "OK Sign",
    "Pointing",
    "Pinch",
    "Swipe Left",
    "Swipe Right",
    "Five Fingers",
    "Call Gesture",
    "Heart Gesture",
    "Brightness Gesture",
    "Two Finger Down"
]
