
import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)
from PIL import Image

def load_model():
    model = MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image):
    img_array = np.array(image)
    img_array = cv2.resize(img_array, (224, 224))
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def classify_image(model, image):
    try:
        preprocessed_image = preprocess_image(image)
        predictions = model.predict(preprocessed_image)
        decoded_predictions = decode_predictions(predictions, top=3)[0]
        return decoded_predictions
    except Exception as e:
        st.error(f"Error during classification: {e}")
        return None
    
def main():
    st.set_page_config(page_title="AI Image Classifier", page_icon="🖼️", layout="centered")

    st.title("AI Image Classification")
    st.write("Upload an image and let Ai tell you what it is!")

    @st.cache_resource
    def load_cached_model():
        return load_model()

    model = load_cached_model()

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            image = st.image(
                uploaded_file, caption='Uploaded Image', use_container_width=True
            )

            if st.button("Classify Image"):
                with st.spinner('Classifying...'):
                    image = Image.open(uploaded_file)
                    predictions = classify_image(model, image)

                    if predictions:
                        st.subheader("Predictions:")
                        for _, label, score in predictions:
                            st.write(f"**{label}**: {score:.2f}")

        except Exception as e:
            st.error(f"Error loading image: {e}")

if __name__ == "__main__":
    main()