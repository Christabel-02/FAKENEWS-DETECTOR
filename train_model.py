"""
train_model.py
Run this file in Google Colab or locally to train and save the models.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv2D, MaxPooling2D, Flatten

# --- Create output folder ---
os.makedirs("models", exist_ok=True)

# ================================
# TEXT MODEL
# ================================
print("📖 Training text model...")

text_model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=500),
    LSTM(64),
    Dense(1, activation="sigmoid")
])

text_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Dummy training data (replace with real preprocessed dataset)
X_train_text = np.random.randint(5000, size=(200, 500))
y_train_text = np.random.randint(2, size=(200,))
text_model.fit(X_train_text, y_train_text, epochs=2, batch_size=16, verbose=1)

text_model.save("models/text_model.h5")
print("✅ Text model saved at models/text_model.h5")

# ================================
# IMAGE MODEL
# ================================
print("\n🖼 Training image model...")

image_model = Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

image_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Dummy training data (replace with real dataset later)
X_train_img = np.random.rand(50, 224, 224, 3)
y_train_img = np.random.randint(2, size=(50,))
image_model.fit(X_train_img, y_train_img, epochs=2, batch_size=8, verbose=1)

image_model.save("models/image_model.h5")
print("✅ Image model saved at models/image_model.h5")

print("\n🎉 Training complete. Models are saved inside /models/")

