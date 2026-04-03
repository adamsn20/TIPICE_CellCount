import streamlit as st
from ultralytics import YOLO
from PIL import Image

# Configure the page
st.set_page_config(page_title="Cell Counter", layout="centered")
st.title("Fluorescent E. coli Bacteria Counter")
st.write("Upload an image to automatically segment and count the cells.")

# Load the YOLO model (cached so it doesn't reload on every interaction)
@st.cache_resource
def load_model():
    # Make sure 'best.pt' is in the same directory as this script
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
            # Run the model
            results = model(image)
            
            # The number of detected instances (cells) is the length of the boxes array
            cell_count = len(results[0].boxes)
            
            # Display the count
            st.success(f"**Detected Cell Count:** {cell_count}")
            
            # Generate the image with the segmentation masks drawn on it
            annotated_image = results[0].plot()
            
            # Display the annotated result
            st.image(annotated_image, caption="Segmentation Results", use_column_width=True)
