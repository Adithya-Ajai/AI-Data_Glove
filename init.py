import serial
import time
import csv
import numpy as np
from collections import deque

PORT = 'COM13'             # Change to your port
BAUD_RATE = 115200
THRESHOLD = 0.5           # Movement threshold (tweak this higher if too sensitive)
WINDOW_TIME = 2.5         # Seconds of data to record once movement is detected
SAMPLE_RATE = 50          # Samples per second (adjust if different)

label = input("Enter the alphabet label (A, B, C): ")
csv_file = f"dataset_{label}.csv"

ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
time.sleep(2)
print("[INFO] Connected to IMU.")

buffer = deque(maxlen=int(WINDOW_TIME * SAMPLE_RATE))
prev_ax, prev_ay, prev_az = 0, 0, 0

with open(csv_file, 'a', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label'])

    try:
        while True:
            line = ser.readline().decode('utf-8').strip()
            if line:
                parts = line.split(',')
                if len(parts) != 6:
                    continue
                
                ax, ay, az, gx, gy, gz = map(float, parts)

                # Check total accel change
                movement = np.sqrt((ax - prev_ax)**2 + (ay - prev_ay)**2 + (az - prev_az)**2)

                if movement > THRESHOLD:
                    buffer.append([ax, ay, az, gx, gy, gz])

                    # If buffer is full, save it
                    if len(buffer) == buffer.maxlen:
                        print(f"[INFO] Movement detected, saving {len(buffer)} samples for {label}")
                        for data in buffer:
                            writer.writerow(data + [label])
                        f.flush()
                        buffer.clear()

                prev_ax, prev_ay, prev_az = ax, ay, az

    except KeyboardInterrupt:
        print("\n[INFO] Stopping data recording.")

ser.close()
