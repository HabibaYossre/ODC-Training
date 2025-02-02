import streamlit as st
from ultralytics import YOLO
import torch
from PIL import Image
import numpy as np
import cv2

# Load YOLOv8 model with Streamlit caching
@st.cache_resource
def load_model():
    return YOLO("yolov8s.pt")  

# Function to analyze the image and return detected components and the image with bounding boxes

def analyze_image(model, image):
    image_np = np.array(image)  
    
    results = model(image_np)[0]  

    labels = model.names  # Class names dictionary

    # Extract class IDs
    class_ids = results.boxes.cls.cpu().numpy().astype(int)  # Convert tensor to numpy

    detected_labels = list(set(labels[class_id] for class_id in class_ids))  # Unique labels

    # Draw bounding boxes on the image
    for box, conf, cls in zip(results.boxes.xyxy.cpu().numpy(), 
                              results.boxes.conf.cpu().numpy(), 
                              class_ids):
        label = f'{labels[cls]} {conf:.2f}'
        plot_one_box(box, image_np, label=label, color=(255, 0, 0), line_thickness=2)

    return detected_labels, Image.fromarray(image_np)  # Convert NumPy array back to PIL Image

# Function to plot bounding boxes on the image

def plot_one_box(box, img, color=(255, 0, 0), label=None, line_thickness=2):
    box = [int(i) for i in box]
    c1, c2 = (box[0], box[1]), (box[2], box[3])
    cv2.rectangle(img, c1, c2, color, thickness=line_thickness, lineType=cv2.LINE_AA)

    if label:
        font_scale = max(0.5, line_thickness / 3)
        font_thickness = max(1, line_thickness // 2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        text_pos = (c1[0], c1[1] - text_size[1] - 3)
        cv2.rectangle(img, (c1[0], c1[1] - text_size[1] - 10), 
                      (c1[0] + text_size[0], c1[1]), color, -1)  # Filled box
        cv2.putText(img, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                    [225, 255, 255], font_thickness, lineType=cv2.LINE_AA)

# Streamlit UI

st.title("YOLOv8 Object Detection App")
st.write("Upload an image and click 'Analyze Image' to detect objects.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # Convert image to RGB
    st.image(image, caption='Uploaded Image', use_container_width=True)

    if st.button("Analyze Image"):
        model = load_model()
        detected_components, detected_image = analyze_image(model, image)

        st.image(detected_image, caption='Detected Image', use_container_width=True)
        st.write("Detected Components:")
        for component in detected_components:
            st.write(f"- {component}")
