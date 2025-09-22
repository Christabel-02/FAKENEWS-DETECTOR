import streamlit as st
from PIL import Image
import numpy as np
import os
import gdown
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("🌐 Fake News Detection (Text + Image)")

# --- Safe directory for models ---
MODEL_DIR = "tmp/model"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Google Drive model IDs (replace with your own) ---
text_model_id = "1uVzRVVMEsdIxw7pItqiWN3RI_Mgp8GRQ"
image_model_id = "1mq6HKFdE2cjNnrhp5f4ROKA4RRCrFxIr"
tokenizer_id = "YOUR_TOKENIZER_FILE_ID"  # 👈 Upload tokenizer.pkl to Drive

text_model_path = os.path.join(MODEL_DIR, "text_model.h5")
image_model_path = os.path.join(MODEL_DIR, "image_model.h5")
tokenizer_path = os.path.join(MODEL_DIR, "tokenizer.pkl")

# --- Download models from Google Drive ---
if not os.path.exists(text_model_path):
    gdown.download(f"https://drive.google.com/uc?id={text_model_id}", text_model_path, quiet=False)

if not os.path.exists(image_model_path):
    gdown.download(f"https://drive.google.com/uc?id={image_model_id}", image_model_path, quiet=False)

if not os.path.exists(tokenizer_path):
    gdown.download(f"https://drive.google.com/uc?id={tokenizer_id}", tokenizer_path, quiet=False)

# --- Load models + tokenizer ---
@st.cache_resource
def load_models():
    text_model = load_model(text_model_path)
    image_model = load_model(image_model_path)
    with open(tokenizer_path, "rb") as f:
        tokenizer = pickle.load(f)
    return text_model, image_model, tokenizer

text_model, image_model, tokenizer = load_models()

# --- Tabs ---
tab1, tab2 = st.tabs(["Text News", "Image News"])

with tab1:
    st.header("📰 Text-based News Detection")
    news_text = st.text_area("Paste news article here:")
    if st.button("Predict Text"):
        if news_text.strip() == "":
            st.warning("Please enter some news text!")
        else:
            seq = tokenizer.texts_to_sequences([news_text])
            padded = pad_sequences(seq, maxlen=500, padding='post', truncating='post')

            pred = text_model.predict(padded)[0][0]
            if pred > 0.5:
                st.error("⚠️ This news is likely **Fake**")
            else:
                st.success("✅ This news is likely **True**")

with tab2:
    st.header("🖼 Image-based News Detection")
    uploaded_file = st.file_uploader("Upload a news image", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        img = image.resize((224, 224))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        pred = image_model.predict(img_array)[0][0]
        if pred > 0.5:
            st.error("⚠️ This news image is likely **Fake**")
        else:
            st.success("✅ This news image is likely **True**")

