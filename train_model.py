"""
train_model.py
Run this file in Google Colab or locally to train and save the models.
"""

import os
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

# --- Create output folder ---
os.makedirs("models", exist_ok=True)

# ================================
# TEXT MODEL (BERT)
# ================================
print("📖 Training BERT text model...")

# Load tokenizer & model
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
text_model = TFBertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Dummy training dataset (replace with real dataset)
texts = [
    "The cat is a mammal",
    "The moon is made of cheese",
    "Dogs are animals",
    "Fish can fly in the sky"
]
labels = [1, 0, 1, 0]  # 1 = True, 0 = Fake

encodings = tokenizer(texts, truncation=True, padding=True, return_tensors="tf", max_length=64)

dataset = tf.data.Dataset.from_tensor_slices((
    dict(encodings),
    labels
)).batch(2)

# Compile & train
text_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=2e-5),
                   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=["accuracy"])

text_model.fit(dataset, epochs=2)

# Save model
text_model.save_pretrained("models/text_model")
tokenizer.save_pretrained("models/text_model")
print("✅ BERT text model saved at models/text_model/")

# ================================
# IMAGE MODEL (CNN)
# ================================
print("\n🖼 Training image model...")

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

image_model = Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

image_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Dummy training data (replace with real dataset)
X_train_img = np.random.rand(20, 224, 224, 3)
y_train_img = np.random.randint(2, size=(20,))
image_model.fit(X_train_img, y_train_img, epochs=2, batch_size=4, verbose=1)

image_model.save("models/image_model.h5")
print("✅ Image model saved at models/image_model.h5")

print("\n🎉 Training complete. Models are saved inside /models/")
