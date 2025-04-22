import serial
import time
import numpy as np
import tensorflow as tf
from collections import deque

PORT = 'COM13'
BAUD_RATE = 115200
THRESHOLD = 0.5
WINDOW_SIZE = 20

model = tf.keras.models.load_model('processed/model.h5')
labels = ['A', 'B', 'C']

ser = serial.Serial(PORT, BAUD_RATE)
time.sleep(2)

window = deque(maxlen=WINDOW_SIZE)

print("[INFO] Listening to IMU...")

while True:
    line = ser.readline().decode().strip()
    parts = line.split(',')
    if len(parts) != 6:
        continue

    ax, ay, az, gx, gy, gz = map(float, parts)

    movement = abs(ax) + abs(ay) + abs(az)
    if movement < THRESHOLD:
        continue

    sample = np.array([ax, ay, az, gx, gy, gz])
    window.append(sample)

    if len(window) == WINDOW_SIZE:
        # Normalize window (0-1 scaling)
        arr = np.array(window)
        min_vals = arr.min(axis=0)
        max_vals = arr.max(axis=0)
        norm_window = (arr - min_vals) / (max_vals - min_vals)

        input_data = norm_window.flatten().reshape(1, -1)
        prediction = model.predict(input_data)
        predicted_label = labels[np.argmax(prediction)]

        print(f"[PREDICTION] {predicted_label}")
        window.clear()

ser.close()
