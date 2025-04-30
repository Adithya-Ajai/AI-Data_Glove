import tensorflow as tf

# === 1. Load your Keras model ===
model = tf.keras.models.load_model('imu_digit_model.h5')
print("[INFO] Loaded imu_digit_model.h5 successfully.")

# === 2. Create the TFLite converter ===
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# === 3. Important settings to handle TensorList ops ===
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Normal TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # TF ops (for unsupported stuff like TensorListReserve)
]
converter._experimental_lower_tensor_list_ops = False  # Don't lower TensorList ops

# Optional: Still enable optimizations
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# === 4. Convert the model ===
tflite_model = converter.convert()
print("[INFO] Converted model to TFLite format.")

# === 5. Save the TFLite model ===
tflite_model_path = 'imu_digit_model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f"[SUCCESS] Saved TFLite model to {tflite_model_path}")
