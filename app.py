import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

st.set_page_config(
    page_title="YOLOv11 Detection",
    layout="centered"
)

st.title("🧠 YOLOv11 Image Object Detection")
st.write("Upload an image and detect objects using YOLOv11")

# Load model (cached)
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

model = load_model()

uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    if st.button("🔍 Detect Objects"):
        with st.spinner("Running YOLO detection..."):
            img_np = np.array(image)
            img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

            # Save temp image
            with tempfile.NamedTemporaryFile(
                suffix=".jpg",
                delete=False
            ) as tmp:
                temp_path = tmp.name
                cv2.imwrite(temp_path, img_bgr)

            # Run YOLO
            results = model(t
