from pathlib import Path
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

MODEL_PATH = Path("waste_model.keras")
CLASS_NAMES = ["organic", "recyclable"]

st.set_page_config(page_title="Smart Waste Segregation", page_icon="♻️")
st.title("AI Smart Waste Segregation Dustbin")
st.write("Upload a waste image to see the model's predicted category.")
st.caption("Student prototype only. Check the prediction manually; this demo does not control a real motor.")

if not MODEL_PATH.exists():
    st.warning("No trained model found yet. Prepare the dataset and run `python train.py` first.")
    st.stop()

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

uploaded = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)
    resized = image.resize((224, 224))
    array = np.asarray(resized, dtype=np.float32) / 255.0
    prediction = load_model().predict(np.expand_dims(array, axis=0), verbose=0)[0]
    index = int(np.argmax(prediction))
    confidence = float(prediction[index])
    st.subheader(f"Predicted category: {CLASS_NAMES[index].title()}")
    st.write(f"Model confidence: {confidence:.1%}")
    if confidence < 0.70:
        st.warning("Confidence is low. Please sort this item manually.")
    st.info("Confidence is not a guarantee of correctness. Test the model with varied real-world images.")
