import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
import pickle

# --- Create output folder ---
os.makedirs("models", exist_ok=True)

# ================================
# TEXT MODEL
# ================================
print("📖 Training text model...")

# Dummy text dataset (replace with your real dataset later)
X_train_text = np.random.randint(5000, size=(200, 500))
y_train_text = np.random.randint(2, size=(200,))

# Define text model
text_model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=500),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

text_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train text model
text_model.fit(X_train_text, y_train_text, epochs=3, batch_size=16, verbose=1)

# Save text model
text_model.save("models/text_model.h5")
print("✅ Text model saved at models/text_model.h5")

# Save tokenizer (dummy tokenizer for now – replace with real preprocessing later)
from tensorflow.keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(["this is a dummy example"])  # 👈 replace with real training texts
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
print("✅ Tokenizer saved at models/tokenizer.pkl")

# ================================
# IMAGE MODEL
# ================================
print("\n🖼 Training image model...")

# Dummy image dataset (replace with your real dataset later)
X_train_img = np.random.rand(50, 224, 224, 3)
y_train_img = np.random.randint(2, size=(50,))

# Define image model
image_model = Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid")
])

image_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train image model
image_model.fit(X_train_img, y_train_img, epochs=3, batch_size=8, verbose=1)

# Save image model
image_model.save("models/image_model.h5")
print("✅ Image model saved at models/image_model.h5")

print("\n🎉 Training complete. Models and tokenizer are saved inside /models/")

