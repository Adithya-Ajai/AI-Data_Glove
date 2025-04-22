import numpy as np
import tensorflow as tf
import os
from tensorflow.keras import layers, models
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# === Config ===
PROCESSED_DIR = 'processed_data'
MODEL_SAVE_PATH = 'gesture_model.h5'

# === Load Data ===
print("[INFO] Loading preprocessed data...")
X_train = np.load(os.path.join(PROCESSED_DIR, 'X_train.npy'))
X_test = np.load(os.path.join(PROCESSED_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(PROCESSED_DIR, 'y_train.npy'), allow_pickle=True)
y_test = np.load(os.path.join(PROCESSED_DIR, 'y_test.npy'), allow_pickle=True)


# === Encode labels ===
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# === Build the Neural Network Model ===
model = models.Sequential()

# Input layer and hidden layers
model.add(layers.InputLayer(input_shape=(X_train.shape[1],)))  # Input shape based on IMU features
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.2))

# Output layer (3 classes for 3 gestures)
model.add(layers.Dense(3, activation='softmax'))  # 3 classes for 3 alphabets

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# === Train the Model ===
print("[INFO] Training model...")
model.fit(X_train, y_train_encoded, epochs=10, batch_size=32, validation_data=(X_test, y_test_encoded))

# === Evaluate the Model ===
print("[INFO] Evaluating model...")
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded, verbose=2)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")

# === Save the Model ===
print("[INFO] Saving model...")
model.save(MODEL_SAVE_PATH)

# === Make Predictions and Show Accuracy ===
y_pred = model.predict(X_test)
y_pred_labels = label_encoder.inverse_transform(np.argmax(y_pred, axis=1))

accuracy = accuracy_score(y_test, y_pred_labels)
print(f"[INFO] Final accuracy: {accuracy * 100:.2f}%")
