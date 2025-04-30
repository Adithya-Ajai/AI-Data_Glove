import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

# Load dataset
data = np.load('imu_dataset.npz')
X = data['X']  # Shape: (samples, 100, 6)
y = data['y']  # Shape: (samples,)

# Normalize inputs (to range -1 to 1)
X = X / np.max(np.abs(X), axis=(1, 2), keepdims=True)

# One-hot encode labels
encoder = LabelBinarizer()
y_encoded = encoder.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Define LSTM model
model = Sequential([
    LSTM(64, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.4),
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')  # 10 digits (0-9)
])

# Compile model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=20, batch_size=16)

# Save model
model.save("imu_digit_model.h5")
print("✅ Model saved as imu_digit_model.h5")

