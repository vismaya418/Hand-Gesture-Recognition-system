import os
import joblib
import numpy as np
import tensorflow as tf
from tensorflow import keras
from src import config
from src.utils import logger, preprocess_landmarks

class GestureClassifier:
    """
    Gesture Classification wrapper.
    Loads the trained neural network model and the label encoder to perform real-time prediction.
    """
    def __init__(self):
        self.model = None
        self.encoder = None
        self.load_model()

    def load_model(self):
        """Loads model and encoder from configured paths."""
        # Load label encoder
        if os.path.exists(config.LABEL_ENCODER_PATH):
            try:
                self.encoder = joblib.load(config.LABEL_ENCODER_PATH)
                logger.info(f"Label encoder loaded from {config.LABEL_ENCODER_PATH}")
            except Exception as e:
                logger.error(f"Failed to load label encoder: {e}")
        else:
            logger.warning(f"Label encoder not found at {config.LABEL_ENCODER_PATH}")

        # Load TensorFlow/Keras model
        if os.path.exists(config.MODEL_PATH):
            try:
                # Load model
                self.model = keras.models.load_model(config.MODEL_PATH)
                logger.info(f"TensorFlow model loaded from {config.MODEL_PATH}")
            except Exception as e:
                logger.error(f"Failed to load Keras model: {e}")
        else:
            logger.warning(f"Keras model not found at {config.MODEL_PATH}")

    def is_ready(self) -> bool:
        """Returns True if both model and encoder are successfully loaded."""
        return self.model is not None and self.encoder is not None

    def predict(self, landmarks) -> tuple[str, float]:
        """
        Predicts gesture from raw landmark coordinates.
        Input: list of 21 landmarks with x, y coordinates.
        Returns:
            predicted_label (str): Name of the gesture, e.g. "Thumbs Up"
            confidence (float): Probability of the gesture [0.0, 1.0]
        """
        if not self.is_ready():
            return "Model Not Loaded", 0.0

        try:
            # 1. Preprocess landmarks to 42 features
            features = preprocess_landmarks(landmarks)
            features_arr = np.expand_dims(features, axis=0)  # Shape (1, 42)

            # 2. Run inference
            predictions = self.model.predict(features_arr, verbose=0)
            class_idx = np.argmax(predictions[0])
            confidence = float(predictions[0][class_idx])

            # 3. Decode label
            predicted_label = self.encoder.classes_[class_idx]
            return predicted_label, confidence
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return "Prediction Error", 0.0
