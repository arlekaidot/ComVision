import cv2
import streamlit as st
import numpy as np
from PIL import Image

with st.sidebar:
    st.title("Thresholding to read unreadable images")
    img = st.file_uploader("Upload Your Image", type=['jpg', 'png', 'jpeg'])
    st.write("Original Image")
    st.image(img)

    adapt_block = st.slider("Block Size Slider for Adaptive Thresholding", min_value = 3, max_value = 300, step = 2, value = 41)
    adapt_c = st.slider("Enter C Size to reduce Noise", value = 5)




images = np.array(Image.open(img))

st.write("Converting to Grayscale")
picture = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
st.image(picture)

st.write("Binary Thresholding")
_,result = cv2.threshold(picture, 20, 255, cv2.THRESH_BINARY)
st.image(result)


st.write("Adaptive Thresholding")
adaptive_result = cv2.adaptiveThreshold(picture, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, adapt_block, adapt_c)
st.image(adaptive_result)

