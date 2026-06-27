import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow import keras
from keras import layers

from src import config
from src.utils import logger, preprocess_landmarks

def generate_mock_landmarks(gesture_name: str) -> list:
    """
    Generates mock raw 21 landmark coordinates for testing.
    Uses basic anatomical approximations + Gaussian noise.
    """
    # Initialize base coordinates (21 landmarks)
    coords = np.zeros((21, 2))
    
    # 0 is wrist
    coords[0] = [0.5, 0.8]
    
    # Define finger extensions
    # 1-4: Thumb, 5-8: Index, 9-12: Middle, 13-16: Ring, 17-20: Pinky
    # Standard finger spacing in X relative to wrist
    fx = [0.2, 0.4, 0.5, 0.6, 0.7] # thumb, index, middle, ring, pinky base X offsets
    
    def set_folded(start, end, base_x, direction=1):
        # folded fingers are curved down towards the palm center
        for i in range(start, end + 1):
            coords[i] = [base_x + (i - start) * 0.02 * direction, 0.5 + (i - start) * 0.03]
            
    def set_extended(start, end, base_x, direction=1, angle_y=-1):
        # extended fingers point upwards/downwards
        for i in range(start, end + 1):
            coords[i] = [base_x + (i - start) * 0.02 * direction, 0.5 + (i - start) * 0.1 * angle_y]

    # Defaults: set all folded
    set_folded(1, 4, 0.4, -1)
    set_folded(5, 8, 0.45)
    set_folded(9, 12, 0.5)
    set_folded(13, 16, 0.55)
    set_folded(17, 20, 0.6)

    # Modify based on gesture
    if gesture_name in ["Open Palm", "Five Fingers"]:
        set_extended(1, 4, 0.35, -1)
        set_extended(5, 8, 0.45)
        set_extended(9, 12, 0.5)
        set_extended(13, 16, 0.55)
        set_extended(17, 20, 0.6)
    elif gesture_name == "Fist":
        # All tightly folded (kept as default)
        pass
    elif gesture_name == "Thumbs Up":
        set_extended(1, 4, 0.3, -1) # thumb extended up
    elif gesture_name == "Thumbs Down":
        set_extended(1, 4, 0.3, -1, angle_y=1) # thumb extended down
    elif gesture_name == "Peace":
        set_extended(5, 8, 0.45) # index extended
        set_extended(9, 12, 0.55) # middle extended
    elif gesture_name == "Pointing":
        set_extended(5, 8, 0.45) # index extended
    elif gesture_name == "OK Sign":
        # Thumb and index tips touch (at index 4 and 8)
        set_extended(1, 3, 0.4, -1)
        set_extended(5, 7, 0.45)
        coords[4] = [0.45, 0.4] # tips touch
        coords[8] = [0.45, 0.4]
        set_extended(9, 12, 0.55)
        set_extended(13, 16, 0.6)
        set_extended(17, 20, 0.65)
    elif gesture_name == "Pinch":
        # Thumb and index tips touch, others folded
        set_extended(1, 3, 0.4, -1)
        set_extended(5, 7, 0.45)
        coords[4] = [0.45, 0.45]
        coords[8] = [0.45, 0.45]
    elif gesture_name == "Swipe Left":
        # Index extended horizontally to the left
        for i in range(5, 9):
            coords[i] = [0.5 - (i - 4) * 0.1, 0.5]
    elif gesture_name == "Swipe Right":
        # Index extended horizontally to the right
        for i in range(5, 9):
            coords[i] = [0.5 + (i - 4) * 0.1, 0.5]
    elif gesture_name == "Call Gesture":
        set_extended(1, 4, 0.3, -1) # thumb extended
        set_extended(17, 20, 0.7) # pinky extended
    elif gesture_name == "Heart Gesture":
        # Single hand mini heart (thumb and index crossed)
        coords[4] = [0.48, 0.42]
        coords[8] = [0.52, 0.4]
    elif gesture_name == "Brightness Gesture":
        # Index and middle extended straight up and close
        set_extended(5, 8, 0.48)
        set_extended(9, 12, 0.52)
    elif gesture_name == "Two Finger Down":
        # Index and middle extended straight down and close
        set_extended(5, 8, 0.48, angle_y=1)
        set_extended(9, 12, 0.52, angle_y=1)

    # Add Gaussian noise
    noise = np.random.normal(0, 0.02, coords.shape)
    coords += noise
    
    # Flatten to list
    return coords.flatten().tolist()

def generate_mock_dataset(filepath: str, num_samples_per_gesture=150):
    """Generates a full mock dataset and saves it to CSV."""
    logger.info("Dataset CSV not found or empty. Generating mock dataset...")
    rows = []
    
    # Header
    header = []
    for i in range(21):
        header.append(f"x_{i}")
        header.append(f"y_{i}")
    header.append("label")
    
    for gesture in config.GESTURES:
        for _ in range(num_samples_per_gesture):
            raw_coords = generate_mock_landmarks(gesture)
            row = raw_coords + [gesture]
            rows.append(row)
            
    df = pd.DataFrame(rows, columns=header)
    df.to_csv(filepath, index=False)
    logger.info(f"Mock dataset generated successfully at {filepath} with {len(df)} samples.")

def main():
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load or Generate Dataset
    if not os.path.exists(config.DATASET_PATH) or os.path.getsize(config.DATASET_PATH) < 100:
        generate_mock_dataset(config.DATASET_PATH, num_samples_per_gesture=150)
        
    logger.info("Loading dataset CSV...")
    df = pd.read_csv(config.DATASET_PATH)
    
    # 2. Extract features and labels
    # Features: columns x_0 to y_20 (42 features)
    feature_cols = [col for col in df.columns if col.startswith('x_') or col.startswith('y_')]
    X_raw = df[feature_cols].values
    y_raw = df['label'].values
    
    logger.info("Normalizing and preprocessing landmark coordinates...")
    # Preprocess all rows to be invariant
    X = []
    for row in X_raw:
        # Reconstruct landmarks list as x, y pairs
        lms = [[row[2*i], row[2*i+1]] for i in range(21)]
        preprocessed = preprocess_landmarks(lms)
        X.append(preprocessed)
        
    X = np.array(X)
    
    # 3. Label Encoding
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    
    # Save the label encoder
    joblib.dump(encoder, config.LABEL_ENCODER_PATH)
    logger.info(f"Saved Label Encoder to {config.LABEL_ENCODER_PATH}")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
    
    # 4. Build Model
    model = keras.Sequential([
        layers.Input(shape=(42,)),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(len(encoder.classes_), activation='softmax')
    ])
    
    # Compile
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    model.summary()
    
    # 5. Train
    epochs = 40
    batch_size = 32
    
    logger.info("Training TensorFlow model...")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(X_test, y_test),
        verbose=1
    )
    
    # Save Model
    model.save(config.MODEL_PATH)
    logger.info(f"Saved TensorFlow model to {config.MODEL_PATH}")
    
    # 6. Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Test Accuracy: {test_acc*100:.2f}%, Test Loss: {test_loss:.4f}")
    
    # Plot accuracy and loss
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    perf_path = config.ASSETS_DIR / "training_performance.png"
    plt.savefig(perf_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved training performance plots to {perf_path}")
    
    # Confusion Matrix
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    cm_path = config.ASSETS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix plot to {cm_path}")
    
    # Print Classification Report
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=encoder.classes_))

if __name__ == "__main__":
    main()
