import serial
import time
import os
import numpy as np
import csv
from collections import deque

PORT = 'COM13'  # Change to your port
BAUD_RATE = 115200
THRESHOLD = 0.5
WINDOW_TIME = 1
SAMPLE_RATE = 50

TEMPLATE_DIR = "templates"
if not os.path.exists(TEMPLATE_DIR):
    os.makedirs(TEMPLATE_DIR)

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print("[INFO] Connected to IMU.")

prev_ax, prev_ay, prev_az = 0, 0, 0

def save_template(data, label, example_num):
    file_path = os.path.join(TEMPLATE_DIR, f"{label}_{example_num}.csv")
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ax', 'ay', 'az', 'gx', 'gy', 'gz'])
        writer.writerows(data)
    print(f"[INFO] Saved {file_path}")

label = input("Enter the alphabet label (A, B, C): ")

example_num = 1
buffer = deque(maxlen=int(WINDOW_TIME * SAMPLE_RATE))

try:
    while True:
        line = ser.readline().decode('utf-8').strip()
        if line:
            parts = line.split(',')
            if len(parts) != 6:
                continue

            ax, ay, az, gx, gy, gz = map(float, parts)

            movement = np.sqrt((ax - prev_ax)**2 + (ay - prev_ay)**2 + (az - prev_az)**2)

            if movement > THRESHOLD:
                buffer.append([ax, ay, az, gx, gy, gz])

                if len(buffer) == buffer.maxlen:
                    print(f"[INFO] Recording template {label}_{example_num}...")
                    save_template(list(buffer), label, example_num)
                    buffer.clear()
                    example_num += 1

            prev_ax, prev_ay, prev_az = ax, ay, az

except KeyboardInterrupt:
    print("\n[INFO] Stopped template recording.")

ser.close()
