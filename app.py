import streamlit as st
from streamlit_drawable_canvas import st_canvas
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2


# Load the pre-trained model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('mnist_model.h5')


model = load_model()

# Streamlit app
st.title("Handwritten Digit Recognition")

# Create a canvas for drawing
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",
    stroke_width=20,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)


# Function to preprocess the canvas image
def preprocess_image(data):
    if data is None or data.image_data is None:
        return None
    # Convert canvas image to PIL Image
    img = Image.fromarray(data.image_data.astype('uint8'), 'RGBA')
    img = img.convert('L')  # Convert to grayscale

    # Apply thresholding to binarize the image
    img = np.array(img)
    _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)

    # Find the bounding box of the digit
    coords = cv2.findNonZero(img)
    if coords is None:
        return None
    x, y, w, h = cv2.boundingRect(coords)
    if w > 0 and h > 0:
        img = img[y:y + h, x:x + w]
    else:
        img = img  # Fallback if no digit is detected

    # Resize to 28x28, preserving aspect ratio with padding
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # Reshape to (1, 28, 28, 1)

    return img


# Predict button
if st.button("Predict Digit"):
    if canvas_result.image_data is not None:
        # Preprocess the drawn image
        processed_img = preprocess_image(canvas_result)
        if processed_img is not None:
            # Make prediction
            prediction = model.predict(processed_img)
            digit = np.argmax(prediction)
            confidence = np.max(prediction) * 100

            st.success(f"Predicted Digit: {digit} (Confidence: {confidence:.2f}%)")

            # Display the processed image
            st.image(processed_img.reshape(28, 28), caption="Processed Image", clamp=True)
        else:
            st.error("Error processing image. Please draw a digit and try again!")
    else:
        st.error("Please draw a digit first!")

# Clear button
if st.button("Clear Canvas"):
    st.session_state.canvas = None
    st.experimental_rerun()