import os
import cv2
import streamlit as st
import numpy as np
from PIL import Image

st.title("Image Detection System using YOLO")

# Get current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

weights_path = os.path.join(BASE_DIR, "yolov3.weights")
config_path = os.path.join(BASE_DIR, "yolov3.cfg")
names_path = os.path.join(BASE_DIR, "coco.names")

net = cv2.dnn.readNet(weights_path, config_path)

with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]
