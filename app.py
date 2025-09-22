import streamlit as st
import numpy as np
import pickle
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from PIL import Image
import os

# ================================
# Google Drive File IDs (replace with yours)
# ================================
TEXT_MODEL_ID = "/1uVzRVVMEsdIxw7pItqiWN3RI_Mgp8GRQ"
IMAGE_MODEL_ID = "1mq6HKFdE2cjNnrhp5f4ROKA4RRCrFxIr"
TOKENIZER_ID  = "1xGHQ_RYMPvc4Vfz4zo7-kaLfiM5WldoZ"

# ================================
# Download files if not present
# ================================
os.makedirs("models", exist_ok=True)

if not os.path.exists("text_model.h5"):
    gdown.download(f"https://drive.google.com/uc?id={TEXT_MODEL_ID}", "models/text_model.h5", quiet=False)

if not os.path.exists("image_model.h5"):
    gdown.download(f"https://drive.google.com/uc?id={IMAGE_MODEL_ID}", "models/image_model.h5", quiet=False)

if not os.path.exists("tokenizer.pkl"):
    gdown.download(f"https://drive.google.com/uc?id={TOKENIZER_ID}", "models/tokenizer.pkl", quiet=False)


# ================================
# Load models and tokenizer
# ================================
@st.cache_resource
def load_text_model():
    return load_model("text_model.h5")

@st.cache_resource
def load_image_model():
    return load_model("image_model.h5")

@st.cache_resource
def load_tokenizer():
    with open("tokenizer.pkl", "rb") as f:
        return pickle.load(f)

text_model = load_text_model()
image_model = load_image_model()
tokenizer = load_tokenizer()

MAXLEN = 500  # must match training


# ================================
# Prediction functions
# ================================
def predict_text(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAXLEN, padding="post", truncating="post")
    pred = text_model.predict(padded)[0][0]
    return "✅ Real News" if pred > 0.5 else "❌ Fake News", float(pred)


def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1,224,224,3)
    pred = image_model.predict(img_array)[0][0]
    return "✅ Real Image" if pred > 0.5 else "❌ Fake Image", float(pred)


# ================================
# Streamlit UI
# ================================
st.set_page_config(page_title="Fake News Detector", page_icon="📰", layout="centered")

st.title("📰 Fake News Detection System")
st.write("Detect fake news using **Text Analysis** and **Image Classification**")

# Tabs for Text and Image
tab1, tab2 = st.tabs(["📝 Text News", "🖼 Image News"])

# --- Text Tab ---
with tab1:
    st.subheader("Enter a news article:")
    user_text = st.text_area("Paste news content here...", height=150)

    if st.button("🔍 Analyze Text"):
        if user_text.strip():
            label, prob = predict_text(user_text)
            st.success(f"**Prediction:** {label}")
            st.info(f"Confidence Score: {prob:.2f}")
        else:
            st.warning("⚠️ Please enter some text!")

# --- Image Tab ---
with tab2:
    st.subheader("Upload a news image:")
    uploaded_img = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

    if uploaded_img is not None:
        image = Image.open(uploaded_img).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

        if st.button("🔍 Analyze Image"):
            label, prob = predict_image(image)
            st.success(f"**Prediction:** {label}")
            st.info(f"Confidence Score: {prob:.2f}")
