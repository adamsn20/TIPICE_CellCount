import streamlit as st
import os

# --- STREAMLIT CLOUD HACK ---
# ultralytics sneaks in the GUI version of OpenCV. 
# This uninstalls it on boot so Python is forced to use the headless version.
@st.cache_resource
def force_opencv_headless():
    os.system("pip uninstall -y opencv-python")

force_opencv_headless()
# ----------------------------

from ultralytics import YOLO
from PIL import Image

# Configure the page
st.set_page_config(page_title="Cell Counter", layout="centered")
st.title("Fluorescent E. coli Bacteria Counter")
st.write("Upload an image to automatically segment and count the cells.")

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Run inference when the user clicks the button
    if st.button("Count Cells"):
        with st.spinner("Analyzing image..."):
            results = model(image)
            cell_count = len(results[0].boxes)
            st.success(f"**Detected Cell Count:** {cell_count}")
            
            annotated_image = results[0].plot()
            st.image(annotated_image, caption="Segmentation Results", use_column_width=True)
