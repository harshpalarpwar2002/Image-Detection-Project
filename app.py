import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Page configuration
st.set_page_config(page_title="YOLO11 Object Detection", layout="wide")

st.title("🚀 YOLO11 Object Detection App")
st.write("Upload an image to see the model in action!")

# Load the model
# Ensure 'yolo11n.pt' is in the same directory as this script
model = YOLO('yolo11n.pt')

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert the file to an image
    image = Image.open(uploaded_file)
    
    # Create two columns for side-by-side view
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Original Image")
        st.image(image, use_container_width=True)

    with col2:
        st.header("Detection Result")
        # Run inference
        results = model(image)
        
        # Plot the results on the image (returns a numpy array)
        annotated_img = results[0].plot()
        
        # Display the annotated image
        st.image(annotated_img, channels="BGR", use_container_width=True)
        
        # Optional: Show detection details
        with st.expander("See detection details"):
            st.write(results[0].probs if results[0].probs else "Objects detected:")
            for box in results[0].boxes:
                st.write(f"Class: {model.names[int(box.cls)]}, Conf: {box.conf[0]:.2f}")
