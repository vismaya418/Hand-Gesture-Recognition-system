import os
import time
import subprocess
import ctypes
import webbrowser
import pyautogui
from datetime import datetime
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import screen_brightness_control as sbc

from src import config
from src.utils import logger, log_action_to_csv, VoiceAssistant

# Disable pyautogui fail-safe to prevent crash at screen corners
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0.05  # Minimal pause for smooth control

class ActionController:
    """
    Executes system actions based on predicted gestures.
    Manages cooldowns, continuous actions, mouse smoothing, and state tracking.
    """
    def __init__(self, voice_assistant: VoiceAssistant):
        self.tts = voice_assistant
        self.last_action_time = {}      # Maps action_name -> timestamp
        self.palm_start_time = None      # For the 3-second palm hold mute
        self.palm_mute_triggered = False
        self.last_gesture = None
        self.gesture_streak = 0
        self.required_gesture_streak = 3
        self.last_triggered_one_shot_gesture = None
        self.one_shot_action_locked = False
        
        # Mouse smoothing variables
        self.prev_x, self.prev_y = pyautogui.position()
        self.smoothing = 0.25            # Exponential moving average factor (lower = smoother)
        
        # Initialize Audio interfaces
        self.volume_interface = self._init_volume_interface()
        self.mic_interface = self._init_mic_interface()

    def _init_volume_interface(self):
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return interface.QueryInterface(IAudioEndpointVolume)
        except Exception as e:
            logger.error(f"Failed to initialize speaker volume interface: {e}")
            return None

    def _init_mic_interface(self):
        try:
            mic = AudioUtilities.GetMicrophone()
            interface = mic.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return interface.QueryInterface(IAudioEndpointVolume)
        except Exception as e:
            logger.error(f"Failed to initialize microphone interface: {e}")
            return None

    def _check_cooldown(self, action_name: str, cooldown_duration: float) -> bool:
        """Returns True if the action is allowed (cooldown elapsed)."""
        now = time.time()
        last_time = self.last_action_time.get(action_name, 0.0)
        if now - last_time >= cooldown_duration:
            self.last_action_time[action_name] = now
            return True
        return False

    def execute_action(self, gesture: str, confidence: float, hand_info: dict = None) -> tuple[str, str]:
        """
        Main entry point for executing action based on prediction.
        Returns a tuple of (action_performed, system_status_message).
        """
        if confidence < config.CONFIDENCE_THRESHOLD and gesture not in ["Pointing", "Pinch"]:
            self._reset_palm_mute_state()
            return "None", "Low confidence gesture ignored"

        action = "None"
        status_msg = "Idle"
        start_time = time.time()

        # -------------------------------------------------------------
        # 1. State Machine: Palm Held for 3 Seconds (Mute Microphone)
        # -------------------------------------------------------------
        if gesture == "Open Palm":
            if self.palm_start_time is None:
                self.palm_start_time = time.time()
                self.palm_mute_triggered = False
            
            elapsed = time.time() - self.palm_start_time
            if elapsed >= 3.0 and not self.palm_mute_triggered:
                action, status_msg = self._toggle_microphone()
                self.palm_mute_triggered = True  # Prevent firing repeatedly during same hold
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg
            elif not self.palm_mute_triggered:
                status_msg = f"Holding Palm for Mute... ({int(3 - elapsed)}s)"
        else:
            self._reset_palm_mute_state()

        # -------------------------------------------------------------
        # 2. Continuous Action: Pointing Finger (Mouse Movement)
        # -------------------------------------------------------------
        if gesture == "Pointing" and hand_info:
            action, status_msg = self._move_mouse(hand_info)
            return action, status_msg

        # -------------------------------------------------------------
        # 3. Continuous Action: Pinch (Mouse Left Click)
        # -------------------------------------------------------------
        elif gesture == "Pinch":
            if self._check_cooldown("Mouse Click", 0.8):
                action, status_msg = self._click_mouse()
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg
            return "Mouse Click", "Clicking (cooldown)"

        # -------------------------------------------------------------
        # 4. Discrete Actions (Volume, Brightness, Navigation, Apps)
        # -------------------------------------------------------------
        elif gesture == "Thumbs Up":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Increase Volume", config.CONTINUOUS_ACTION_COOLDOWN):
                action, status_msg = self._change_volume(0.05)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "Thumbs Down":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Decrease Volume", config.CONTINUOUS_ACTION_COOLDOWN):
                action, status_msg = self._change_volume(-0.05)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "Brightness Gesture":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Increase Brightness", config.CONTINUOUS_ACTION_COOLDOWN):
                action, status_msg = self._change_brightness(10)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "Two Finger Down":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Decrease Brightness", config.CONTINUOUS_ACTION_COOLDOWN):
                action, status_msg = self._change_brightness(-10)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "Peace":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Take Screenshot", config.DISCRETE_ACTION_COOLDOWN):
                action, status_msg = self._take_screenshot()
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "OK Sign":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Open Calculator", config.DISCRETE_ACTION_COOLDOWN):
                action, status_msg = self._open_calculator()
                self._record_one_shot_gesture(gesture)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "Five Fingers":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Open Chrome", config.DISCRETE_ACTION_COOLDOWN):
                action, status_msg = self._open_browser("https://www.google.com")
                self._record_one_shot_gesture(gesture)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "Call Gesture":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Open WhatsApp Web", config.DISCRETE_ACTION_COOLDOWN):
                action, status_msg = self._open_browser("https://web.whatsapp.com")
                self._record_one_shot_gesture(gesture)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "Heart Gesture":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Open Spotify", config.DISCRETE_ACTION_COOLDOWN):
                action, status_msg = self._open_browser("https://open.spotify.com")
                self._record_one_shot_gesture(gesture)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "Swipe Right":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Next Slide", config.DISCRETE_ACTION_COOLDOWN):
                action, status_msg = self._navigate_slide("right")
                self._record_one_shot_gesture(gesture)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "Swipe Left":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Previous Slide", config.DISCRETE_ACTION_COOLDOWN):
                action, status_msg = self._navigate_slide("left")
                self._record_one_shot_gesture(gesture)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        elif gesture == "Fist":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Lock Workstation", 5.0): # High cooldown for locking screen
                action, status_msg = self._lock_computer()
                self._record_one_shot_gesture(gesture)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        # Handle fallback for Open Palm gesture if it hasn't hit 3s yet
        elif gesture == "Open Palm":
            if self._should_trigger_discrete_action(gesture) and self._check_cooldown("Play/Pause Media", config.DISCRETE_ACTION_COOLDOWN):
                action, status_msg = self._play_pause_media()
                self._record_one_shot_gesture(gesture)
                duration = time.time() - start_time
                log_action_to_csv(gesture, confidence, action, duration)
                return action, status_msg

        return "None", "Gesture detected but in cooldown"

    def _reset_palm_mute_state(self):
        self.palm_start_time = None
        self.palm_mute_triggered = False

    def _should_trigger_discrete_action(self, gesture: str) -> bool:
        """Require a short streak of the same gesture before firing discrete actions."""
        if not hasattr(self, "gesture_streak") or not hasattr(self, "last_gesture"):
            self.last_gesture = None
            self.gesture_streak = 0
        if not hasattr(self, "one_shot_action_locked"):
            self.one_shot_action_locked = False
        if not hasattr(self, "last_triggered_one_shot_gesture"):
            self.last_triggered_one_shot_gesture = None

        if gesture != self.last_gesture:
            self.last_gesture = gesture
            self.gesture_streak = 1
            if gesture != self.last_triggered_one_shot_gesture:
                self.one_shot_action_locked = False
            return False

        if gesture == self.last_triggered_one_shot_gesture and self.one_shot_action_locked:
            return False

        self.gesture_streak += 1
        required_streak = getattr(self, "required_gesture_streak", 3)
        if not hasattr(self, "required_gesture_streak"):
            self.required_gesture_streak = required_streak
        return self.gesture_streak >= required_streak

    def _record_one_shot_gesture(self, gesture: str):
        self.last_triggered_one_shot_gesture = gesture
        self.one_shot_action_locked = True

    # ==========================================
    # Action Implementations
    # ==========================================

    def _move_mouse(self, hand_info: dict) -> tuple[str, str]:
        # Landmark 8 is Index Finger Tip
        # hand_info['lm_list'] format: [id, px_x, px_y, norm_x, norm_y, norm_z]
        lm_8 = hand_info['lm_list'][8]
        norm_x, norm_y = lm_8[3], lm_8[4]
        
        # Screen dimensions
        screen_w, screen_h = pyautogui.size()
        
        # Crop coordinate workspace so small hand movement spans full screen
        # e.g., mapping normalized X: [0.25, 0.75] -> [0, screenWidth]
        min_x, max_x = 0.25, 0.75
        min_y, max_y = 0.25, 0.75
        
        # Clamp normalized coords
        clamped_x = max(min_x, min(max_x, norm_x))
        clamped_y = max(min_y, min(max_y, norm_y))
        
        # Interpolate
        target_x = int((clamped_x - min_x) / (max_x - min_x) * screen_w)
        target_y = int((clamped_y - min_y) / (max_y - min_y) * screen_h)
        
        # Apply exponential moving average for smoothing
        curr_x = int(self.smoothing * target_x + (1 - self.smoothing) * self.prev_x)
        curr_y = int(self.smoothing * target_y + (1 - self.smoothing) * self.prev_y)
        
        pyautogui.moveTo(curr_x, curr_y)
        self.prev_x, self.prev_y = curr_x, curr_y
        
        return "Mouse Move", f"Cursor at ({curr_x}, {curr_y})"

    def _click_mouse(self) -> tuple[str, str]:
        pyautogui.click()
        return "Mouse Click", "Performed Left Mouse Click"

    def _change_volume(self, change_amt: float) -> tuple[str, str]:
        if not self.volume_interface:
            return "Volume Control", "Speaker volume interface not available"
        try:
            current_vol = self.volume_interface.GetMasterVolumeLevelScalar()
            new_vol = min(1.0, max(0.0, current_vol + change_amt))
            self.volume_interface.SetMasterVolumeLevelScalar(new_vol, None)
            pct = int(new_vol * 100)
            
            # Speak only on discrete intervals or volume changes (handled by caller or helper)
            if change_amt > 0 and self._check_cooldown("Voice Volume Inc", 2.0):
                self.tts.speak("Volume Increased")
            elif change_amt < 0 and self._check_cooldown("Voice Volume Dec", 2.0):
                self.tts.speak("Volume Decreased")
                
            return "Volume Control", f"Volume set to {pct}%"
        except Exception as e:
            logger.error(f"Error adjusting volume: {e}")
            return "Volume Control", "Error adjusting speaker volume"

    def _change_brightness(self, change_amt: int) -> tuple[str, str]:
        try:
            current = sbc.get_brightness()
            if isinstance(current, list):
                current = current[0]
            new_bright = min(100, max(0, current + change_amt))
            sbc.set_brightness(new_bright)
            
            if change_amt > 0 and self._check_cooldown("Voice Bright Inc", 2.0):
                self.tts.speak("Brightness Increased")
            elif change_amt < 0 and self._check_cooldown("Voice Bright Dec", 2.0):
                self.tts.speak("Brightness Decreased")
                
            return "Brightness Control", f"Brightness set to {new_bright}%"
        except Exception as e:
            logger.warning(f"Error adjusting brightness: {e}")
            return "Brightness Control", f"Brightness adjustment failed ({e})"

    def _toggle_microphone(self) -> tuple[str, str]:
        if not self.mic_interface:
            return "Microphone Control", "Microphone interface not available"
        try:
            is_muted = self.mic_interface.GetMute()
            new_state = not is_muted
            self.mic_interface.SetMute(new_state, None)
            
            msg = "Microphone Muted" if new_state else "Microphone Unmuted"
            self.tts.speak(msg)
            return "Microphone Toggle", msg
        except Exception as e:
            logger.error(f"Error toggling microphone: {e}")
            return "Microphone Control", "Error toggling microphone mute state"

    def _play_pause_media(self) -> tuple[str, str]:
        pyautogui.press("playpause")
        return "Play/Pause Media", "Toggled media play/pause state"

    def _take_screenshot(self) -> tuple[str, str]:
        try:
            config.SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"screenshot_{timestamp}.png"
            path = config.SCREENSHOTS_DIR / filename
            
            # Perform screenshot
            screenshot = pyautogui.screenshot()
            screenshot.save(path)
            
            self.tts.speak("Screenshot Taken")
            return "Take Screenshot", f"Saved to screenshots/{filename}"
        except Exception as e:
            logger.error(f"Screenshot failure: {e}")
            return "Take Screenshot", f"Screenshot failed: {e}"

    def _open_calculator(self) -> tuple[str, str]:
        try:
            subprocess.Popen("calc.exe", shell=True)
            self.tts.speak("Calculator Opened")
            return "Open Calculator", "Calculator launched successfully"
        except Exception as e:
            logger.error(f"Failed to open calculator: {e}")
            return "Open Calculator", f"Failed: {e}"

    def _open_browser(self, url: str) -> tuple[str, str]:
        try:
            webbrowser.open(url)
            self.tts.speak("Browser Opened")
            return "Open Browser", f"Opened URL: {url}"
        except Exception as e:
            logger.error(f"Failed to open URL {url}: {e}")
            return "Open Browser", f"Failed: {e}"

    def _navigate_slide(self, direction: str) -> tuple[str, str]:
        key = "right" if direction == "right" else "left"
        pyautogui.press(key)
        self.tts.speak("Presentation Started" if self._check_cooldown("Voice Presentation", 5.0) else None)
        return "Slide Navigation", f"Pressed {key} arrow key"

    def _lock_computer(self) -> tuple[str, str]:
        try:
            self.tts.speak("System Locking")
            time.sleep(0.5)  # Let TTS start speaking
            ctypes.windll.user32.LockWorkStation()
            return "Lock PC", "Locked Windows session"
        except Exception as e:
            logger.error(f"Failed to lock workstation: {e}")
            return "Lock PC", f"Failed: {e}"
