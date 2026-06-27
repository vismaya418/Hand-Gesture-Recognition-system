import os
import time
import urllib.request
import cv2
import mediapipe as mp
from src import config
from src.utils import logger

try:
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    MP_TASKS_AVAILABLE = True
except Exception as exc:
    mp_python = None
    mp_vision = None
    MP_TASKS_AVAILABLE = False
    MP_TASKS_IMPORT_ERROR = exc

try:
    from mediapipe.solutions import hands as mp_hands
    MP_LEGACY_AVAILABLE = True
except Exception as exc:
    mp_hands = None
    MP_LEGACY_AVAILABLE = False
    MP_LEGACY_IMPORT_ERROR = exc

# Canonical hand connection joint pairs
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (5, 6), (6, 7), (7, 8),              # Index
    (9, 10), (10, 11), (11, 12),          # Middle
    (13, 14), (14, 15), (15, 16),        # Ring
    (17, 18), (18, 19), (19, 20),        # Pinky
    (0, 5), (5, 9), (9, 13), (13, 17), (0, 17) # Palm/Knuckle boundaries
]

class HandDetector:
    """
    Wrapper class around MediaPipe Tasks HandLandmarker.
    Handles hand detection, drawing landmarks, and coordinate extraction.
    Bypasses legacy mp.solutions which is missing on certain environments (e.g. Python 3.13 custom wheels).
    """
    def __init__(self, mode=False, max_hands=2, complexity=1, detection_con=0.5, track_con=0.5):
        self.mode = mode
        self.max_hands = max_hands
        self.complexity = complexity
        self.detection_con = detection_con
        self.track_con = track_con
        self.legacy_mode = False
        self.results = None

        if MP_TASKS_AVAILABLE:
            # Path to hand landmarker task model
            self.model_path = os.path.join(config.MODELS_DIR, "hand_landmarker.task")
            
            # Download task model if not present locally
            if not os.path.exists(self.model_path):
                try:
                    logger.info("hand_landmarker.task model file is missing. Downloading from Google CDN...")
                    url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
                    urllib.request.urlretrieve(url, self.model_path)
                    logger.info(f"Model downloaded and saved to {self.model_path}")
                except Exception as e:
                    logger.error(f"Failed to download MediaPipe task model: {e}")
                    raise RuntimeError(f"Could not load MediaPipe task model asset: {e}")
            
            # Initialize HandLandmarker Options
            base_options = mp_python.BaseOptions(model_asset_path=self.model_path)
            options = mp_vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_con,
                min_hand_presence_confidence=detection_con,
                min_tracking_confidence=track_con
            )
            self.detector = mp_vision.HandLandmarker.create_from_options(options)
        elif MP_LEGACY_AVAILABLE:
            self.detector = mp_hands.Hands(
                static_image_mode=mode,
                max_num_hands=max_hands,
                min_detection_confidence=detection_con,
                min_tracking_confidence=track_con,
            )
            self.legacy_mode = True
        else:
            raise RuntimeError(
                "MediaPipe could not be initialized. Install a compatible mediapipe package in the active Python environment. "
                f"Task import error: {MP_TASKS_IMPORT_ERROR if 'MP_TASKS_IMPORT_ERROR' in globals() else 'n/a'}; "
                f"Legacy import error: {MP_LEGACY_IMPORT_ERROR if 'MP_LEGACY_IMPORT_ERROR' in globals() else 'n/a'}"
            )

    def find_hands(self, img, draw=True):
        """
        Detects hands in a BGR image and optionally draws landmarks and connections using OpenCV.
        """
        # Convert BGR to RGB (MediaPipe requirement)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.legacy_mode:
            self.results = self.detector.process(img_rgb)
            if self.results and self.results.multi_hand_landmarks and draw:
                h, w, _ = img.shape
                for hand_lms in self.results.multi_hand_landmarks:
                    for connection in HAND_CONNECTIONS:
                        p1 = hand_lms.landmark[connection[0]]
                        p2 = hand_lms.landmark[connection[1]]
                        cv2.line(
                            img,
                            (int(p1.x * w), int(p1.y * h)),
                            (int(p2.x * w), int(p2.y * h)),
                            (0, 255, 0), 2
                        )
                    for lm in hand_lms.landmark:
                        cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)
            return img

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Execute detection
        self.results = self.detector.detect(mp_image)
        
        if self.results and self.results.hand_landmarks and draw:
            h, w, _ = img.shape
            for hand_lms in self.results.hand_landmarks:
                # 1. Draw connections (lines)
                for connection in HAND_CONNECTIONS:
                    p1 = hand_lms[connection[0]]
                    p2 = hand_lms[connection[1]]
                    cv2.line(
                        img, 
                        (int(p1.x * w), int(p1.y * h)), 
                        (int(p2.x * w), int(p2.y * h)), 
                        (0, 255, 0), 2
                    )
                # 2. Draw joints (circles)
                for lm in hand_lms:
                    cv2.circle(img, (int(lm.x * w), int(lm.y * h)), 4, (0, 0, 255), -1)
                    
        return img

    def get_hands_list(self, img, draw=True):
        """
        Extracts landmark coordinates for all detected hands.
        Returns a list of dictionaries containing:
        - 'lm_list': list of landmarks with [id, px_x, px_y, norm_x, norm_y, norm_z]
        - 'bbox': [xmin, ymin, xmax, ymax] in pixel coordinates
        - 'type': 'Left' or 'Right' hand classification
        """
        hands_list = []
        if self.legacy_mode:
            if not self.results or not self.results.multi_hand_landmarks:
                return hands_list
            h, w, _ = img.shape
            for idx, hand_lms in enumerate(self.results.multi_hand_landmarks):
                lm_list = []
                x_list = []
                y_list = []
                hand_type = "Right"
                if self.results.multi_handedness and idx < len(self.results.multi_handedness):
                    hand_type = self.results.multi_handedness[idx].classification[0].label
                for lm_id, lm in enumerate(hand_lms.landmark):
                    px_x, px_y = int(lm.x * w), int(lm.y * h)
                    lm_list.append([lm_id, px_x, px_y, lm.x, lm.y, lm.z])
                    x_list.append(px_x)
                    y_list.append(px_y)
                if x_list and y_list:
                    xmin, xmax = min(x_list), max(x_list)
                    ymin, ymax = min(y_list), max(y_list)
                    bbox = [xmin, ymin, xmax, ymax]
                    if draw:
                        cv2.rectangle(img, (xmin - 10, ymin - 10), (xmax + 10, ymax + 10), (255, 0, 255), 2)
                        cv2.putText(img, hand_type, (xmin - 10, ymin - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
                    hands_list.append({
                        'lm_list': lm_list,
                        'bbox': bbox,
                        'type': hand_type
                    })
            return hands_list

        if not self.results or not self.results.hand_landmarks:
            return hands_list
            
        h, w, _ = img.shape
        
        for idx, hand_lms in enumerate(self.results.hand_landmarks):
            lm_list = []
            x_list = []
            y_list = []
            
            # Determine handedness label (Left/Right)
            hand_type = "Right"
            if self.results.handedness and idx < len(self.results.handedness):
                # category_name returns 'Left' or 'Right'
                hand_type = self.results.handedness[idx][0].category_name
                
            for lm_id, lm in enumerate(hand_lms):
                px_x, px_y = int(lm.x * w), int(lm.y * h)
                # Format: [id, px_x, px_y, norm_x, norm_y, norm_z]
                lm_list.append([lm_id, px_x, px_y, lm.x, lm.y, lm.z])
                x_list.append(px_x)
                y_list.append(px_y)
                
            # Compute hand bounding box
            if x_list and y_list:
                xmin, xmax = min(x_list), max(x_list)
                ymin, ymax = min(y_list), max(y_list)
                bbox = [xmin, ymin, xmax, ymax]
                
                if draw:
                    cv2.rectangle(img, (xmin - 10, ymin - 10), (xmax + 10, ymax + 10), (255, 0, 255), 2)
                    cv2.putText(img, hand_type, (xmin - 10, ymin - 20), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 255), 2)
                    
                hands_list.append({
                    'lm_list': lm_list,
                    'bbox': bbox,
                    'type': hand_type
                })
                
        return hands_list

    def close(self):
        """Releases the MediaPipe detector instance resources."""
        if hasattr(self, "detector") and self.detector:
            try:
                self.detector.close()
            except Exception:
                pass


def main():
    """Simple test run of the hand tracker."""
    cap = cv2.VideoCapture(0)
    detector = HandDetector(max_hands=2)
    p_time = 0
    
    logger.info("Starting hand detector test. Press 'q' to quit.")
    
    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                logger.warning("Webcam frame not available.")
                break
                
            img = cv2.flip(img, 1)  # Mirror frame
            img = detector.find_hands(img)
            hands = detector.get_hands_list(img)
            
            if hands:
                # Print wrist info for the first hand
                first_hand = hands[0]
                wrist = first_hand['lm_list'][0]
                cv2.putText(img, f"Wrist px: ({wrist[1]}, {wrist[2]}) Hand: {first_hand['type']}", 
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time
            
            cv2.putText(img, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("Hand Tracker Test", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()

if __name__ == "__main__":
    main()
