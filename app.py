import json
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model

IMG_SIZE = 224
MODEL_PATH = "/app/models/best_model.h5"
CLASS_NAMES_PATH = "/app/models/class_names.json"
LOW_CONFIDENCE = 0.70


@st.cache_resource
def load_resources():
    model = load_model(MODEL_PATH, compile=False)
    with open(CLASS_NAMES_PATH) as f:
        class_names = json.load(f)
    return model, class_names


def preprocess(image_bgr):
    image = cv2.resize(image_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    image = image.astype("float32") / 255.0
    return np.expand_dims(image, axis=0)


st.set_page_config(page_title="BrainScan AI")
st.title("BrainScan AI")
st.write("Upload a brain MRI image to classify it as glioma, meningioma, pituitary or no tumor.")


model, class_names = load_resources()

uploaded = st.file_uploader("Choose an MRI image", type=["jpg", "jpeg", "png", "bmp"])

if uploaded is not None:
    file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
    image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if image_bgr is None:
        st.error("Could not read this file as an image.")
    else:
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="Uploaded image", width=350)

        with st.spinner("Analyzing..."):
            probs = model.predict(preprocess(image_bgr), verbose=0)[0]

        idx = int(np.argmax(probs))
        st.subheader(f"Prediction: {class_names[idx]}")
        st.write(f"Confidence: {probs[idx] * 100:.1f}%")

        if probs[idx] < LOW_CONFIDENCE:
            st.info("Low confidence: this case needs review by a specialist.")

        st.bar_chart({name: float(p) for name, p in zip(class_names, probs)})
        st.caption("A high confidence does not guarantee a correct result.")