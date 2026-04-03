import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Configure the page
st.set_page_config(page_title="Cell Counter", layout="centered")
st.title("Fluorescent E. coli Bacteria Counter")
st.write("Upload an image to automatically segment and count the cells.")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "tif", "tiff"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    if st.button("Count Cells"):
        with st.spinner("Analyzing image..."):
            results = model(image)
            cell_count = len(results[0].boxes)
            st.success(f"**Detected Cell Count:** {cell_count}")
            
            annotated_image = results[0].plot()
            st.image(annotated_image, caption="Segmentation Results", use_column_width=True)
