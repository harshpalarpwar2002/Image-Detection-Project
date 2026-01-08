import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import tempfile
import os

st.set_page_config(page_title="YOLOv11 Detection")

st.title("🧠 YOLOv11 Image Object Detection")

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
    st.image(image, use_container_width=True)

    if st.button("Detect Objects"):
        with st.spinner("Detecting..."):
            img = np.array(image)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                cv2.imwrite(tmp.name, img)
                results = model(tmp.name)

            os.remove(tmp.name)

            annotated = results[0].plot()
            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            st.image(annotated, use_container_width=True)

            st.subheader("Detected Objects")
            for box in results[0].boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                st.write(f"**{model.names[cls]}** — {conf:.2f}")
