import os
import time
import cv2
import pandas as pd
import streamlit as st
import screen_brightness_control as sbc
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

from src.hand_detector import HandDetector
from src.gesture_classifier import GestureClassifier
from src.action_controller import ActionController
from src.utils import logger, VoiceAssistant
from src import config

# Page Config
st.set_page_config(
    page_title="AI Hand Gesture Automation Hub",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Glassmorphic Dark CSS Injection
st.markdown("""
    <style>
        /* General background */
        .reportview-container {
            background: #0e1117;
        }
        .main {
            background-color: #0d0f14;
            color: #f0f2f6;
        }
        
        /* Glassmorphic Cards */
        .metric-card {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(255, 255, 255, 0.05);
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            margin-bottom: 15px;
            text-align: center;
        }
        
        .metric-label {
            font-size: 0.85rem;
            color: #8892b0;
            text-transform: uppercase;
            letter-spacing: 1.5px;
            margin-bottom: 8px;
        }
        
        .metric-value {
            font-size: 1.8rem;
            font-weight: 700;
            color: #64ffda;
        }
        
        .metric-sub {
            font-size: 0.75rem;
            color: #52607a;
            margin-top: 5px;
        }
        
        /* Action Highlight Card */
        .action-card {
            background: linear-gradient(135deg, rgba(100, 255, 218, 0.1) 0%, rgba(0, 0, 0, 0.3) 100%);
            border-radius: 12px;
            padding: 20px;
            border: 1px solid rgba(100, 255, 218, 0.2);
            text-align: center;
            margin-bottom: 15px;
        }
        
        /* Status pulse indicator */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 8px;
        }
        .status-active {
            background-color: #00e676;
            box-shadow: 0 0 10px #00e676;
        }
        .status-idle {
            background-color: #ff9100;
            box-shadow: 0 0 10px #ff9100;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #e2e8f0 !important;
            font-weight: 600 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Helper function to get current speaker volume
def get_system_volume():
    try:
        devices = AudioUtilities.GetSpeakers()
        interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)
        return int(volume.GetMasterVolumeLevelScalar() * 100)
    except Exception:
        return 0

# Helper function to get current screen brightness
def get_system_brightness():
    try:
        b = sbc.get_brightness()
        if isinstance(b, list):
            b = b[0]
        return int(b)
    except Exception:
        return 0

# Helper function to get microphone mute status
def get_system_mic_mute():
    try:
        mic = AudioUtilities.GetMicrophone()
        interface = mic.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = interface.QueryInterface(IAudioEndpointVolume)
        return "Muted" if volume.GetMute() else "Unmuted"
    except Exception:
        return "Unknown"

# Helper function to load latest logs
def load_latest_logs(n=10):
    if not os.path.exists(config.GESTURE_LOG_PATH):
        return pd.DataFrame(columns=["Date", "Time", "Gesture", "Confidence", "Action", "Duration"])
    try:
        df = pd.read_csv(config.GESTURE_LOG_PATH)
        return df.tail(n).iloc[::-1]  # Return last n logs, reversed (newest first)
    except Exception:
        return pd.DataFrame()

# Helper function to get total gestures executed today
def get_today_total_gestures():
    if not os.path.exists(config.GESTURE_LOG_PATH):
        return 0, 0
    try:
        df = pd.read_csv(config.GESTURE_LOG_PATH)
        today = time.strftime("%Y-%m-%d")
        today_df = df[df["Date"] == today]
        total = len(today_df)
        screenshots = len(today_df[today_df["Action"].str.contains("Screenshot", na=False)])
        return total, screenshots
    except Exception:
        return 0, 0

# Initialize Session State Variables
if "running" not in st.session_state:
    st.session_state.running = False
if "voice" not in st.session_state:
    st.session_state.voice = None
if "classifier" not in st.session_state:
    st.session_state.classifier = None
if "detector" not in st.session_state:
    st.session_state.detector = None
if "controller" not in st.session_state:
    st.session_state.controller = None

# ==========================================
# Sidebar Configuration Panel
# ==========================================
st.sidebar.markdown("## 🎛️ Control Panel")

# System Activation Toggle
if st.sidebar.button("🚀 Start Recognition Engine", use_container_width=True, disabled=st.session_state.running):
    st.session_state.running = True
    # Lazy load engines to prevent memory leaks and threading errors
    if not st.session_state.voice:
        st.session_state.voice = VoiceAssistant()
    if not st.session_state.classifier:
        st.session_state.classifier = GestureClassifier()
    if not st.session_state.detector:
        st.session_state.detector = HandDetector(max_hands=1, detection_con=0.7, track_con=0.7)
    if not st.session_state.controller:
        st.session_state.controller = ActionController(st.session_state.voice)
    
    # Check if classifier successfully loaded model
    if not st.session_state.classifier.is_ready():
        st.sidebar.error("TensorFlow Model missing! Please run 'train_model.py' first.")
        st.session_state.running = False

if st.sidebar.button("🛑 Stop Recognition Engine", use_container_width=True, disabled=not st.session_state.running):
    st.session_state.running = False

st.sidebar.divider()

# Hyperparameter Adjustments
st.sidebar.markdown("### ⚙️ Settings")
conf_thresh = st.sidebar.slider(
    "Confidence Threshold", 
    min_value=0.5, max_value=0.99, 
    value=config.CONFIDENCE_THRESHOLD, 
    step=0.01
)
config.CONFIDENCE_THRESHOLD = conf_thresh

mouse_smooth = st.sidebar.slider(
    "Mouse Movement Smoothing", 
    min_value=0.05, max_value=0.9, 
    value=0.25, 
    step=0.05,
    help="Lower value means smoother movement but slightly more latency."
)

st.sidebar.divider()

# System Diagnostics
st.sidebar.markdown("### 📊 Engine Diagnostics")
status_label = "Active" if st.session_state.running else "Idle"
status_class = "status-active" if st.session_state.running else "status-idle"
st.sidebar.markdown(
    f'<div><span class="status-indicator {status_class}"></span>Engine Status: <b>{status_label}</b></div>', 
    unsafe_allow_html=True
)

st.sidebar.info(
    "💡 Tip: Ensure your hand is within 1-2 meters of the webcam, in a well-lit room for best results."
)

# ==========================================
# Main Dashboard Layout
# ==========================================
st.markdown("<h1 style='text-align: center;'>💻 AI-Powered Hand Gesture Automation Hub</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8892b0;'>Control your Windows computer in real time using advanced computer vision and neural networks.</p>", unsafe_allow_html=True)

# Grid Layout
col_webcam, col_metrics = st.columns([3, 2])

# Left Column - Webcam Feed
with col_webcam:
    st.subheader("📹 Live Camera Stream")
    webcam_placeholder = st.empty()
    if not st.session_state.running:
        # Display a stylish placeholder when system is offline
        webcam_placeholder.image(
            "https://images.unsplash.com/photo-1618005182384-a83a8bd57fbe?auto=format&fit=crop&w=800&q=80", 
            caption="Inference engine offline. Press 'Start' in the sidebar to begin streaming.",
            use_container_width=True
        )

# Right Column - Real-time metrics
with col_metrics:
    st.subheader("⚡ Real-Time Analytics")
    metrics_placeholder = st.empty()

# Bottom Section - History Logs
st.divider()
st.subheader("📜 Event Log & History")
history_placeholder = st.empty()

# Initial render of metrics placeholder while offline
def draw_static_metrics(current_gesture="None", confidence=0.0, action="None", status="Offline", fps=0.0):
    total_gestures, screenshot_count = get_today_total_gestures()
    volume_level = get_system_volume()
    brightness_level = get_system_brightness()
    mic_status = get_system_mic_mute()
    
    with metrics_placeholder.container():
        # Action highlight
        st.markdown(f"""
            <div class="action-card">
                <div class="metric-label">Last Executed Action</div>
                <div class="metric-value" style="color: #64ffda; font-size: 2.2rem;">{action}</div>
                <div class="metric-sub">{status}</div>
            </div>
        """, unsafe_allow_html=True)
        
        # Grid of cards
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Detected Gesture</div>
                    <div class="metric-value" style="color: #64ffda;">{current_gesture}</div>
                    <div class="metric-sub">Conf: {confidence * 100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Total Gestures Today</div>
                    <div class="metric-value">{total_gestures}</div>
                    <div class="metric-sub">Since midnight</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Speaker Volume</div>
                    <div class="metric-value">{volume_level}%</div>
                </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">System FPS</div>
                    <div class="metric-value" style="color: #00e676;">{int(fps)}</div>
                    <div class="metric-sub">Target: 25-30 FPS</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Screenshots Taken</div>
                    <div class="metric-value">{screenshot_count}</div>
                    <div class="metric-sub">Saved in screenshots/</div>
                </div>
            """, unsafe_allow_html=True)
            st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Display Brightness</div>
                    <div class="metric-value">{brightness_level}%</div>
                </div>
            """, unsafe_allow_html=True)
            
        # Micro Mute indicator
        st.markdown(f"""
            <div class="metric-card" style="padding: 10px; margin-top: 5px;">
                <div class="metric-label" style="font-size: 0.75rem;">Microphone State: <b>{mic_status}</b></div>
            </div>
        """, unsafe_allow_html=True)

# Draw initial state
draw_static_metrics()
history_placeholder.dataframe(load_latest_logs(), use_container_width=True)

# ==========================================
# Main Video Capture & Inference Loop
# ==========================================
if st.session_state.running:
    # Setup instances
    detector = st.session_state.detector
    classifier = st.session_state.classifier
    controller = st.session_state.controller
    
    # Adjust controller sensitivity
    controller.smoothing = mouse_smooth
    
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        st.error(f"Could not open webcam at index {config.CAMERA_INDEX}. Verify your camera connection.")
        st.session_state.running = False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        
        p_time = 0
        current_action = "None"
        status_msg = "Running"
        
        logger.info("Streamlit Recognition Loop started.")
        
        # Continuous frame grab
        while st.session_state.running:
            success, img = cap.read()
            if not success:
                logger.warning("Empty camera frame.")
                break
                
            img = cv2.flip(img, 1)  # Mirror
            
            # Detect Hands
            img = detector.find_hands(img, draw=True)
            hands = detector.get_hands_list(img, draw=True)
            
            gesture_name = "None"
            confidence = 0.0
            
            if hands:
                hand = hands[0]
                lm_list = hand['lm_list']
                
                # Classify
                gesture_name, confidence = classifier.predict(lm_list)
                
                # Execute Action
                current_action, status_msg = controller.execute_action(
                    gesture=gesture_name,
                    confidence=confidence,
                    hand_info=hand
                )
            else:
                controller._reset_palm_mute_state()
                
            # Calculate FPS
            c_time = time.time()
            fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
            p_time = c_time
            
            # Draw overlay on image for Streamlit display
            cv2.putText(img, f"Gesture: {gesture_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, f"Conf: {confidence * 100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(img, f"Action: {current_action}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display Image in Streamlit
            # Convert BGR (OpenCV) to RGB (Streamlit)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            webcam_placeholder.image(img_rgb, channels="RGB", use_container_width=True)
            
            # Update metrics cards
            draw_static_metrics(
                current_gesture=gesture_name,
                confidence=confidence,
                action=current_action,
                status=status_msg,
                fps=fps
            )
            
            # Update history logs occasionally (only when action is performed or log CSV changes)
            history_placeholder.dataframe(load_latest_logs(10), use_container_width=True)
            
            # Tiny sleep to yield processor control
            time.sleep(0.01)
            
        # Clean up camera when loop terminates
        cap.release()
        logger.info("Streamlit Recognition Loop stopped. Camera released.")
        st.rerun()  # Rerun to draw the offline screen state
