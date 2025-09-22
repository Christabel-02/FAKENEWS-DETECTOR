import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle

# --- Create output folder ---
os.makedirs("models", exist_ok=True)

# ================================
# TEXT MODEL + TOKENIZER
# ================================
print("📖 Training text model...")

# ----------------
# 1. Tokenizer
# ----------------
# Replace this dummy corpus with your real text dataset
dummy_corpus = [
    "This is real news",
    "This is fake news",
    "Breaking news today",
    "Another real news article",
    "Fake news example"
]

# Initialize tokenizer
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(dummy_corpus)

# Save tokenizer
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# ----------------
# 2. Prepare sequences
# ----------------
sequences = tokenizer.texts_to_sequences(dummy_corpus)
X_text = pad_sequences(sequences, maxlen=500, padding="post", truncating="post")
y_text = np.array([1, 0, 1, 1, 0])  # Example labels: 1=real, 0=fake

# ----------------
# 3. Build model
# ----------------
text_model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=500),
    LSTM(64),
    Dense(1, activation="sigmoid")
])

text_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ----------------
# 4. Train model
# ----------------
text_model.fit(
    X_text, y_text,
    epochs=10,
    batch_size=2,
    validation_split=0.2,
    verbose=1
)

# ----------------
# 5. Save model
# ----------------
text_model.save("models/text_model.h5")
print("✅ Text model + tokenizer saved in models/")

# ================================
# IMAGE MODEL
# ================================
print("\n🖼 Training image model...")

# ----------------
# 1. Build model
# ----------------
image_model = Sequential([
    tf.keras.layers.Input(shape=(224, 224, 3)),
    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation="relu"),
    Dense(1, activation="sigmoid")
])

image_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ----------------
# 2. Dummy training data
# ----------------
# Replace with real images (preprocessed/resized to 224x224)
X_image = np.random.rand(10, 224, 224, 3)  # small dummy dataset
y_image = np.random.randint(2, size=(10,))

# ----------------
# 3. Train model
# ----------------
image_model.fit(
    X_image, y_image,
    epochs=5,
    batch_size=2,
    validation_split=0.2,
    verbose=1
)

# ----------------
# 4. Save model
# ----------------
image_model.save("models/image_model.h5")
print("✅ Image model saved in models/")

print("\n🎉 Training complete. Upload 'text_model.h5', 'image_model.h5', and 'tokenizer.pkl' to Google Drive.")

