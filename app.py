import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

st.title("YOLO Image Classification App")

# Load YOLO model safely
try:
    model = YOLO("yolo11n-cls.pt")
    model.to("cpu")
except Exception as e:
    st.error("Error loading YOLO model. Make sure yolo11n-cls.pt exists in repo.")
    st.stop()

uploaded_file = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        results = model(temp.name)

    class_id = results[0].probs.top1
    class_name = results[0].names[class_id]

    st.success(f"Prediction: {class_name}")
