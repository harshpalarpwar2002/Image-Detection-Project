import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile

# Page configuration
st.set_page_config(page_title="YOLO Object Detection", layout="centered")

st.title("🔍 YOLO Object Detection App")
st.write("Upload an image and detect objects using YOLO model")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

# Image upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert image for YOLO
    image_np = np.array(image)

    # Perform detection
    results = model(image_np)

    # Plot results
    annotated_img = results[0].plot()

    # Convert BGR to RGB
    annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

    st.image(annotated_img, caption="Detected Objects", use_container_width=True)
