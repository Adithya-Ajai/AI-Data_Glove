import serial
import time
import numpy as np
import tensorflow as tf
import pyautogui  # For mouse control
import keyboard  # For key press detection

# Parameters
PORT = 'COM13'  # update this if needed
BAUD = 115200
TIMESTEPS = 100
FEATURES = 6
RECORD_DURATION = 2.5  # seconds

# Load model
model = tf.keras.models.load_model("imu_digit_model.h5")

# Connect to serial
ser = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)
print("🔌 Connected to IMU")

# Menu to choose functionality
print("Press 'Q' to use IMU as a mouse.")
print("Press 'D' to run the digit prediction model.")

while True:
    if keyboard.is_pressed('q'):
        print("🖱️ IMU Mouse Mode Activated. Press 'ESC' to exit.")
        while True:
            try:
                line = ser.readline().decode().strip()
                parts = list(map(float, line.split(',')))
                if len(parts) == FEATURES:
                    # Assuming parts[0] and parts[1] are used for mouse movement
                    x_movement = int(parts[0] * 50)  # Scale as needed
                    y_movement = int(parts[1] * 50)  # Scale as needed
                    pyautogui.moveRel(x_movement, y_movement)
                if keyboard.is_pressed('esc'):
                    print("🛑 Exiting IMU Mouse Mode.")
                    break
            except:
                continue

    elif keyboard.is_pressed('d'):
        print("🤖 Digit Prediction Mode Activated.")
        while True:
            input("👉 Press Enter to record gesture...")
            print("⏺️  Recording for 2.5 seconds...")

            data = []
            start_time = time.time()
            while time.time() - start_time < RECORD_DURATION:
                try:
                    line = ser.readline().decode().strip()
                    parts = list(map(float, line.split(',')))
                    if len(parts) == FEATURES:
                        data.append(parts)
                except:
                    continue

            print(f"📦 Collected {len(data)} samples")

            # Convert to numpy
            data = np.array(data)

            # Pad or truncate to TIMESTEPS
            if data.shape[0] < TIMESTEPS:
                pad = np.zeros((TIMESTEPS - data.shape[0], FEATURES))
                data = np.vstack((data, pad))
            elif data.shape[0] > TIMESTEPS:
                data = data[:TIMESTEPS]

            # Normalize
            data = data / np.max(np.abs(data))

            # Add batch dim
            data = np.expand_dims(data, axis=0)

            # Predict
            prediction = model.predict(data)
            digit = np.argmax(prediction)

            print(f" Predicted Digit: {digit}\n")
            if keyboard.is_pressed('esc'):
                print("🛑 Exiting Digit Prediction Mode.")
                break
