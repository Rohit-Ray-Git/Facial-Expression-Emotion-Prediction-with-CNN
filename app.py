import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import base64

# Function to read and encode the image file
def get_base64_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return encoded_string

# Set the background image using CSS
def set_background(image_base64):
    page_bg_img = f"""
    <style>
    .stApp {{
        background: url("data:image/png;base64,{image_base64}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        color: white;  /* Default text color */
    }}
    /* Main content styling */
    .css-1g8v9l0 {{
        background: rgba(255, 255, 255, 0.8); /* Slightly transparent for better readability */
        padding: 20px; /* Padding for content */
        border-radius: 10px; /* Rounded corners */
    }}
    /* Set text color to white for all text elements */
    h1, h2, h3, h4, h5, h6, p, span, div, label {{
        color: white; /* Ensuring all text is white */
    }}
    .stButton > button {{
        background-color: #4C4C6D; /* Button color */
        color: white; /* Text color for button */
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
    }}
    .stButton > button:hover {{
        background-color: #6A5ACD; /* Button hover color */
        color: white;
    }}
    .stSlider > div {{
        background-color: transparent;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the saved model
model = tf.keras.models.load_model('models/image_classifier.h5')

# Set page layout
st.set_page_config(page_title="Mood Prediction", layout="centered")

# CSS for centering and title styling
st.markdown(
    """
    <style>
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #f54033;
    }
    .centered {
        display: flex;
        justify-content: center;
        text-align: center;
        color: #A52A2A;
    }

    .subtitle{
        display: flex;
        justify-content: center;
        text-align: center;
        color: #FF8C00
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.markdown("<h1 class='title'>Mood Prediction App</h1>", unsafe_allow_html=True)

# Centered description text
st.markdown("<h3 class='subtitle'>Upload an image to predict the mood based on the facial expression.</h3>", unsafe_allow_html=True)

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Display the uploaded image if available
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)  # Use use_container_width instead

# Button for making prediction
if st.button("Predict Mood"):
    if uploaded_file is not None:
        # Preprocess the image for prediction
        img_array = np.array(image)
        img_resized = cv2.resize(img_array, (256, 256)) / 255.0
        img_expanded = np.expand_dims(img_resized, axis=0)

        # Make prediction
        prediction = model.predict(img_expanded)[0][0]
        mood = "Happy 😊" if prediction < 0.5 else "Sad 😢"
        
        # Display the result
        st.markdown(f"<h2 class='centered'>Prediction: {mood}</h2>", unsafe_allow_html=True)
    else:
        st.markdown("<h3 class='centered'>Please upload an image first.</h3>", unsafe_allow_html=True)

else:
    st.markdown("<h3 class='centered'>Click the 'Predict Mood' button to make a prediction.</h3>", unsafe_allow_html=True)

# Call the function with the uploaded background image
image_base64 = get_base64_image("img4.jpg")  # Path to your uploaded image (ensure this file exists)
set_background(image_base64)
