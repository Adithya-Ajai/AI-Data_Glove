import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

WINDOW_SIZE = 20

data = pd.read_csv('data/data.csv')
print("[INFO] Loaded data.")

# Normalize (min-max scaling per axis in bounding box)
for col in ['ax', 'ay', 'az', 'gx', 'gy', 'gz']:
    min_val = data[col].min()
    max_val = data[col].max()
    data[col] = (data[col] - min_val) / (max_val - min_val)

# Group by labels, window each
windows = []
labels = []

for label in data['label'].unique():
    group = data[data['label'] == label]
    for start in range(0, len(group) - WINDOW_SIZE + 1, WINDOW_SIZE):
        window = group.iloc[start:start+WINDOW_SIZE][['ax','ay','az','gx','gy','gz']].values.flatten()
        windows.append(window)
        labels.append(label)

X = np.array(windows)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

np.save('processed/X_train.npy', X_train)
np.save('processed/y_train.npy', y_train)
np.save('processed/X_test.npy', X_test)
np.save('processed/y_test.npy', y_test)

print(f"[INFO] Processed {len(X)} samples.")
