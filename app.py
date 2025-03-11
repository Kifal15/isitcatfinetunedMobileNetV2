import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load the trained model
@st.cache_resource()
def load_trained_model():
    return load_model("cat_dog_classifier.h5")  #    return load_model("cat_dog_classifier.h5")  # Ensure you have this model file
 
model = load_trained_model()
IMG_SIZE = (160, 160)  # Change this based on your model input

# Function to preprocess an image
def preprocess_image(img):
    img = img.resize(IMG_SIZE)  # Resize to model input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

# Function for batch prediction
def predict_images(images):
    results = []
    for img in images:
        img_array = preprocess_image(img)
        prediction = model.predict(img_array)
        label = "Cat" if prediction[0][0] < 0.5 else "Dog"
        results.append(label)
    return results

# Streamlit UI
st.title("🐱🐶 Cat vs. Dog Classifier")
st.write("Upload images to classify them as a cat or a dog!")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    images = [Image.open(io.BytesIO(file.read())) for file in uploaded_files]
    predictions = predict_images(images)
    
    for img, label in zip(images, predictions):
        st.image(img, caption=f"Prediction: {label}", use_column_width=True)
        st.title(f"It is a CUTE LIL {label}")