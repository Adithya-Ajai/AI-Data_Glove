import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

X_train = np.load('processed/X_train.npy')
y_train = np.load('processed/y_train.npy')
X_test = np.load('processed/X_test.npy')
y_test = np.load('processed/y_test.npy')

# Label encoding
encoder = LabelEncoder()
y_train_enc = to_categorical(encoder.fit_transform(y_train))
y_test_enc = to_categorical(encoder.transform(y_test))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train_enc, epochs=30, validation_data=(X_test, y_test_enc))
model.save('processed/model.h5')

print("[INFO] Model saved.")
