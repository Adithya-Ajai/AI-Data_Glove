import serial
import time
import numpy as np
import matplotlib.pyplot as plt

PORT = 'COM13'  # Change to your port
BAUD = 115200
LABEL = input("Enter digit label (0-9): ")
DURATION = 2.5  # seconds of capture

ser = serial.Serial(PORT, BAUD)
time.sleep(2)
print("Recording in 2 seconds...")
time.sleep(2)

data = []
start_time = time.time()
while time.time() - start_time < DURATION:
    line = ser.readline().decode().strip()
    try:
        parts = list(map(float, line.split(',')))
        if len(parts) == 6:  # ax, ay, az, gx, gy, gz
            data.append(parts)
    except:
        continue

ser.close()

data = np.array(data)
np.save(f"imu_digit_{LABEL}_{int(time.time())}.npy", data)

plt.plot(data[:, 0], label="Accel X")
plt.plot(data[:, 1], label="Accel Y")
plt.plot(data[:, 2], label="Accel Z")
plt.legend()
plt.title(f"IMU Capture for Digit {LABEL}")
plt.show()
