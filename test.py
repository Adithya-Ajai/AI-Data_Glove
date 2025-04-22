import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
import os
import serial
import time

# === Config ===
MODEL_PATH = 'gesture_model.h5'  # Path to the trained model
PORT = 'COM13'  # Replace with your correct port
BAUD_RATE = 115200
IMU_DATA_LENGTH = 6  # Number of features (ax, ay, az, gx, gy, gz)
STABLE_THRESHOLD = 0.05  # Threshold to detect significant movement
STABLE_TIME_LIMIT = 3  # 3 seconds to recognize a gesture

# === Load the Model ===
print("[INFO] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# === Setup Serial Port for IMU ===
print("[INFO] Setting up serial port...")
ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)  # Allow time for the serial connection to be established

# === Initialize Label Encoder ===
label_encoder = LabelEncoder()
label_encoder.fit(['A', 'B', 'C'])  # Adjust based on your actual labels used during training

# === Read IMU Data and Predict Gesture ===
print("[INFO] Ready to predict gestures...")

# Variables for tracking motion
last_imu_data = None
stable_time = 0
is_moving = False

def read_imu_data():
    # Read IMU data from serial and parse it
    line = ser.readline().decode('utf-8').strip()
    if line:
        imu_values = line.split(',')
        if len(imu_values) == IMU_DATA_LENGTH:
            # Convert the values to float
            imu_data = [float(value) for value in imu_values]
            return np.array(imu_data).reshape(1, -1)  # Reshape for model input
    return None

def is_significant_movement(current_data, last_data):
    """Check if the movement is significant (change in IMU readings above threshold)."""
    if last_data is None:
        return False
    diff = np.abs(current_data - last_data)
    return np.any(diff > STABLE_THRESHOLD)

while True:
    imu_data = read_imu_data()
    
    if imu_data is not None:
        # If there's a significant movement, start the timer and reset stable_time
        if is_significant_movement(imu_data, last_imu_data):
            if not is_moving:
                print("[INFO] Movement detected, starting timer...")
                stable_time = time.time()
                is_moving = True
        
        # If we're in motion, check the time limit
        if is_moving:
            elapsed_time = time.time() - stable_time
            if elapsed_time >= STABLE_TIME_LIMIT:
                # Predict the gesture
                prediction = model.predict(imu_data)
                predicted_label = label_encoder.inverse_transform(np.argmax(prediction, axis=1))
                
                # Output the predicted label
                print(f"Predicted Gesture: {predicted_label[0]}")
                # Reset movement detection
                is_moving = False
        
        # Update last_imu_data
        last_imu_data = imu_data
