import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Traffic Sign Classifier", page_icon="🚦")
st.title("🚦 Smart Traffic Sign Recognition")
st.write("Upload an image of a traffic sign, and the ResNet50V2 AI will classify it.")

CLASS_NAMES = {
    0: "30km/h", 1: "50km/h", 2: "60km/h", 3: "70km/h",
    4: "Yield", 5: "Stop", 6: "No Entry",
    7: "General Caution", 8: "Road Work", 9: "Keep Right"
}

@st.cache_resource
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('traffic_resnet_random.h5')

try:
    model = load_model()
    st.success("AI Model loaded successfully!")
except Exception as e:
    st.error(f"Error loading model: {e}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Traffic Sign.', width=300)
    
    st.write("Processing image...")
    
    image = image.convert('RGB')
    
    img_resized = image.resize((128, 128))
    
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    
    img_array = np.expand_dims(img_array, axis=0)
    
    img_processed = tf.keras.applications.resnet_v2.preprocess_input(img_array)
    
    predictions = model.predict(img_processed)
    predicted_class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    st.markdown("### Results:")
    st.markdown(f"**Prediction:** {CLASS_NAMES[predicted_class_idx]}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")