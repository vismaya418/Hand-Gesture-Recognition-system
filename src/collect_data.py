import os
import csv
import cv2
import pandas as pd
from src.hand_detector import HandDetector
from src import config
from src.utils import logger

def count_samples(label_name: str) -> int:
    """Counts how many samples are already collected in the CSV for a given label."""
    if not os.path.exists(config.DATASET_PATH):
        return 0
    try:
        df = pd.read_csv(config.DATASET_PATH)
        if "label" in df.columns:
            return int((df["label"] == label_name).sum())
    except Exception as e:
        logger.error(f"Error reading dataset CSV: {e}")
    return 0

def select_gesture() -> str:
    """Console prompt to select which gesture to record."""
    print("\n" + "=" * 50)
    print("      SELECT GESTURE TO RECORD")
    print("=" * 50)
    for idx, gesture in enumerate(config.GESTURES):
        count = count_samples(gesture)
        print(f"[{idx:2d}] {gesture:<25} ({count}/300 samples)")
    print("=" * 50)
    
    while True:
        try:
            choice = input(f"Select gesture index (0-{len(config.GESTURES)-1}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                return "quit"
            idx = int(choice)
            if 0 <= idx < len(config.GESTURES):
                return config.GESTURES[idx]
        except ValueError:
            pass
        print(f"Invalid selection. Enter a number between 0 and {len(config.GESTURES)-1}.")

def main():
    # Make sure target dataset directory exists
    config.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check if header needs to be written to CSV
    csv_file_exists = os.path.exists(config.DATASET_PATH)
    if not csv_file_exists:
        try:
            with open(config.DATASET_PATH, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                header = []
                for i in range(21):
                    header.append(f"x_{i}")
                    header.append(f"y_{i}")
                header.append("label")
                writer.writerow(header)
            logger.info(f"Initialized dataset CSV at {config.DATASET_PATH}")
        except Exception as e:
            logger.error(f"Failed to initialize dataset CSV: {e}")
            return

    # Select the first gesture
    current_gesture = select_gesture()
    if current_gesture == "quit":
        logger.info("Exiting data collector.")
        return

    # Initialize Hand Detector
    detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.7)
    
    # Open camera
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error(f"Webcam (index {config.CAMERA_INDEX}) not found.")
        print(f"\nERROR: Webcam index {config.CAMERA_INDEX} could not be opened. Please verify your connection.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    logger.info("Starting collection loop. Press 'S' to save, 'N' to select a different gesture, 'Q' to quit.")
    
    sample_count = count_samples(current_gesture)
    continuous_mode = False
    
    while True:
        success, img = cap.read()
        if not success:
            logger.warning("Failed to grab camera frame.")
            break
            
        img = cv2.flip(img, 1)  # Mirror frame
        
        # Detect hand
        img = detector.find_hands(img, draw=True)
        hands = detector.get_hands_list(img, draw=True)
        
        hand_detected = len(hands) > 0
        status_text = "HAND DETECTED" if hand_detected else "NO HAND DETECTED"
        status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
        
        # Display information on frame
        cv2.putText(img, f"Gesture: {current_gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Samples: {sample_count}/300", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Status: {status_text}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(img, "Press: S=Save, C=Toggle AutoSave, N=Change, Q=Quit", (10, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if continuous_mode:
            cv2.putText(img, "AUTOSAVE ACTIVE", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
        cv2.imshow("Gesture Data Collector", img)
        key = cv2.waitKey(1) & 0xFF
        
        # Handle saving logic
        should_save = False
        if key == ord('s') or key == ord('S'):
            should_save = True
        elif key == ord('c') or key == ord('C'):
            continuous_mode = not continuous_mode
            logger.info(f"Continuous autosave mode: {continuous_mode}")
        elif continuous_mode and hand_detected:
            should_save = True
            
        if should_save:
            if hand_detected:
                # Save first hand landmarks
                hand = hands[0]
                lm_list = hand['lm_list']
                
                # Write to CSV
                try:
                    with open(config.DATASET_PATH, mode="a", newline="", encoding="utf-8") as f:
                        writer = csv.writer(f)
                        row = []
                        # lm_list features: [id, px_x, px_y, norm_x, norm_y, norm_z]
                        # We save normalized x and y coordinates
                        for lm in lm_list:
                            row.append(lm[3]) # norm_x
                            row.append(lm[4]) # norm_y
                        row.append(current_gesture)
                        writer.writerow(row)
                    sample_count += 1
                    # Log progress occasionally
                    if sample_count % 50 == 0 or not continuous_mode:
                        logger.info(f"Collected {sample_count} samples for {current_gesture}")
                except Exception as e:
                    logger.error(f"Error saving sample to CSV: {e}")
            else:
                if not continuous_mode:
                    logger.warning("No hand detected, cannot save sample.")
                    
        # Change gesture
        if key == ord('n') or key == ord('N'):
            cap.release()
            cv2.destroyAllWindows()
            continuous_mode = False
            current_gesture = select_gesture()
            if current_gesture == "quit":
                logger.info("Exiting data collector.")
                return
            sample_count = count_samples(current_gesture)
            # Reopen camera
            cap = cv2.VideoCapture(config.CAMERA_INDEX)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            
        # Quit
        if key == ord('q') or key == ord('Q'):
            logger.info("Quitting data collector.")
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
