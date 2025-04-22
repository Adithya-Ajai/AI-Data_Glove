import serial
import time
import pandas as pd

PORT = 'COM13'
BAUD_RATE = 115200
THRESHOLD = 0.5  # Adjust sensitivity
INTERVAL = 0.1  # seconds

ser = serial.Serial(PORT, BAUD_RATE)
time.sleep(2)
print("[INFO] Connected to IMU")

df = pd.DataFrame(columns=['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'label'])

try:
    while True:
        line = ser.readline().decode().strip()
        parts = line.split(',')
        if len(parts) != 6:
            continue

        ax, ay, az, gx, gy, gz = map(float, parts)

        # Check for significant movement
        movement = abs(ax) + abs(ay) + abs(az)
        if movement < THRESHOLD:
            continue

        label = input("Enter label (A/B/C or q to quit): ")
        if label.lower() == 'q':
            break

        new_row = pd.DataFrame([{'ax': ax, 'ay': ay, 'az': az, 'gx': gx, 'gy': gy, 'gz': gz, 'label': label}])
        df = pd.concat([df, new_row], ignore_index=True)

        print(f"Recorded {label}.")

        time.sleep(INTERVAL)

finally:
    ser.close()
    df.to_csv('data/data.csv', index=False)
    print("[INFO] Data saved to data/data.csv")
