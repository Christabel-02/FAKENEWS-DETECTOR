import streamlit as st
from PIL import Image
import numpy as np
import os
import gdown
import tensorflow as tf
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, BertForSequenceClassification, get_scheduler
from transformers.optimization import AdamW
import torch

st.set_page_config(page_title="Fake News Detection", layout="centered")
st.title("🌐 Fake News Detection (Text + Image)")

# --- Safe directory for models ---
MODEL_DIR = "tmp/model"
os.makedirs(MODEL_DIR, exist_ok=True)

# --- Google Drive model IDs (replace with your IDs) ---
text_model_id = "1uVzRVVMEsdIxw7pItqiWN3RI_Mgp8GRQ"   # replace with your uploaded BERT model zip
image_model_id = "1mq6HKFdE2cjNnrhp5f4ROKA4RRCrFxIr" # replace with your uploaded CNN model h5

text_model_path = os.path.join(MODEL_DIR, "text_model")
image_model_path = os.path.join(MODEL_DIR, "image_model.h5")

# --- Download models from Google Drive ---
if not os.path.exists(text_model_path):
    gdown.download(f"https://drive.google.com/uc?id={text_model_id}", "bert_model.zip", quiet=False)
    os.system("unzip bert_model.zip -d tmp/model/")  # unzip BERT model

if not os.path.exists(image_model_path):
    gdown.download(f"https://drive.google.com/uc?id={image_model_id}", image_model_path, quiet=False)

# --- Load models ---
@st.cache_resource
def load_models():
    # Load BERT (text model) - PyTorch
    text_model = BertForSequenceClassification.from_pretrained(text_model_path)
    tokenizer = BertTokenizer.from_pretrained(text_model_path)
    text_model.eval()  # set to eval mode

    # Load CNN (image model) - Keras
    image_model = load_model(image_model_path)
    return text_model, tokenizer, image_model

text_model, tokenizer, image_model = load_models()

# --- Tabs ---
tab1, tab2 = st.tabs(["Text News", "Image News"])

# ---------------- TEXT TAB ----------------
with tab1:
    st.header("📰 Text-based News Detection")
    news_text = st.text_area("Paste news article here:")
    if st.button("Predict Text"):
        if news_text.strip() == "":
            st.warning("Please enter some news text!")
        else:
            inputs = tokenizer(news_text, return_tensors="pt", padding=True, truncation=True, max_length=64)
            with torch.no_grad():
                outputs = text_model(**inputs)
                probs = torch.nn.functional.softmax(outputs.logits, dim=-1).numpy()[0]
                pred_class = np.argmax(probs)

            if pred_class == 1:  # fake
                st.error(f"⚠️ This news is likely **Fake** (confidence: {probs[1]:.2f})")
            else:  # real
                st.success(f"✅ This news is likely **True** (confidence: {probs[0]:.2f})")

# ---------------- IMAGE TAB ----------------
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

