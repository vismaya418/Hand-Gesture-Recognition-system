import cv2
import time
import sys
from src.hand_detector import HandDetector
from src.gesture_classifier import GestureClassifier
from src.action_controller import ActionController
from src.utils import logger, VoiceAssistant
from src import config

def main():
    logger.info("Initializing Real-Time Prediction System...")
    
    # 1. Initialize background Voice Assistant
    voice = VoiceAssistant()
    
    # 2. Initialize classifier
    classifier = GestureClassifier()
    if not classifier.is_ready():
        logger.error("Model or Label Encoder is missing. Please run 'python -m src.train_model' first.")
        voice.speak("Model files missing. Please train the system first.")
        sys.exit(1)
        
    # 3. Initialize detector and controller
    detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.7)
    controller = ActionController(voice)
    
    # 4. Open webcam
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        logger.error(f"Cannot open webcam index {config.CAMERA_INDEX}")
        voice.speak("Cannot find webcam. System shutting down.")
        sys.exit(1)
        
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    
    p_time = 0
    current_action = "None"
    status_msg = "System Ready"
    
    logger.info("System fully loaded. Press 'q' in the window to quit.")
    voice.speak("System initialized and ready.")
    
    try:
        while cap.isOpened():
            success, img = cap.read()
            if not success:
                logger.warning("Blank frame received from camera.")
                break
                
            img = cv2.flip(img, 1)  # Mirror frame
            
            # Detect Hands
            img = detector.find_hands(img, draw=True)
            hands = detector.get_hands_list(img, draw=True)
            
            gesture_name = "None"
            confidence = 0.0
            
            if hands:
                # Get the first hand info
                hand = hands[0]
                lm_list = hand['lm_list']
                
                # Classify the gesture
                gesture_name, confidence = classifier.predict(lm_list)
                
                # Execute action
                current_action, status_msg = controller.execute_action(
                    gesture=gesture_name,
                    confidence=confidence,
                    hand_info=hand
                )
            else:
                # Reset palm hold states in action controller if no hand detected
                controller._reset_palm_mute_state()
                
            # Compute FPS
            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time
            
            # Display Overlay Info
            cv2.rectangle(img, (5, 5), (320, 140), (0, 0, 0), -1)
            cv2.rectangle(img, (5, 5), (320, 140), (0, 255, 255), 1)
            
            cv2.putText(img, f"Gesture: {gesture_name}", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(img, f"Confidence: {confidence * 100:.1f}%", (15, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            cv2.putText(img, f"Action: {current_action}", (15, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.putText(img, f"FPS: {int(fps)}", (15, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            
            # Display Status Bar at bottom
            cv2.rectangle(img, (0, config.FRAME_HEIGHT - 30), (config.FRAME_WIDTH, config.FRAME_HEIGHT), (30, 30, 30), -1)
            cv2.putText(img, f"Status: {status_msg}", (10, config.FRAME_HEIGHT - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            cv2.imshow("Hand Gesture Controller", img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Closing Predict loop and releasing resources...")
        cap.release()
        cv2.destroyAllWindows()
        voice.stop()
        logger.info("Cleanup complete. Goodbye!")

if __name__ == "__main__":
    main()
