import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import load_model

st.title("Handwritten Digit Classifier")

# Load trained Keras model
model = tf.keras.models.load_model("mnistmodel.keras")

# Upload image widget
file = st.file_uploader("Upload an image of a digit", type=["png", "jpg", "jpeg"])

if file is not None:
    # Display the uploaded image
    img = Image.open(file)
    st.image(img, caption="Uploaded Image", width=150)
    
    # Preprocessing
    img_array = np.array(img.convert("L"))
    if np.mean(img_array) > 127:
        img_array = cv2.bitwise_not(img_array)
    
    img_resized = cv2.resize(img_array, (28, 28))
    img_input = (img_resized / 255.0).reshape(1, 28, 28)
    
    # Predict digit
    if st.button("Predict"):
        pred = model.predict(img_input)
        result = np.argmax(pred)
        st.header(f"Predicted Digit: {result}")