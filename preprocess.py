import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# === Config ===
DATA_FILE = 'imu_dataset.csv'     # raw dataset file
PROCESSED_DIR = 'processed_data'  # output folder
TEST_SIZE = 0.2                   # 20% test split
RANDOM_STATE = 42

# === Load Data ===
print("[INFO] Loading dataset...")
data = pd.read_csv(DATA_FILE)

# Example structure:
# ax, ay, az, gx, gy, gz, label

# Separate features and labels
X = data[['Ax', 'Ay', 'Az', 'Gx', 'Gy', 'Gz']].values
y = data['Label'].values

# === Normalize Features ===
print("[INFO] Scaling data...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Split Train/Test ===
print("[INFO] Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# === Save Processed Data ===
if not os.path.exists(PROCESSED_DIR):
    os.makedirs(PROCESSED_DIR)

np.save(os.path.join(PROCESSED_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(PROCESSED_DIR, 'X_test.npy'), X_test)
np.save(os.path.join(PROCESSED_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(PROCESSED_DIR, 'y_test.npy'), y_test)

print("[INFO] Data preprocessing complete!")
print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
