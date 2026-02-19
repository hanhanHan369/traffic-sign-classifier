import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Vision | Traffic Classifier", layout="centered")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Montserrat:wght@300;400;500&display=swap');

    .stApp {
        background-color: #F4F2EB; 
        color: #4A443B;
        font-family: 'Montserrat', sans-serif;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #4A443B;
        text-align: center;
        font-weight: 400;
        letter-spacing: 1px;
    }

    p {
        font-family: 'Montserrat', sans-serif;
        text-align: center;
        color: #6B655C;
    }

    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF;
        border: 1px solid #D1CDC4;
        border-radius: 2px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.03);
    }

    .stButton>button {
        background-color: #A3A58E; 
        color: white;
        border-radius: 25px;
        border: none;
        padding: 10px 25px;
        font-family: 'Montserrat', sans-serif;
        font-weight: 500;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        font-size: 0.85rem;
        transition: all 0.3s ease;
        display: block;
        margin: 0 auto; 
    }
    
    .stButton>button:hover {
        background-color: #8C8E77; 
        color: white;
    }
    
    [data-testid="stImage"] img {
        border-radius: 4px;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.08);
    }
    
    .stSuccess {
        background-color: #E8EAE3 !important;
        color: #4A443B !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1>Vision - Traffic Sign Classifier </h1>", unsafe_allow_html=True)
st.markdown("<p>Welcome to Vision, our ResNet architecture will seamlessly classify your uploaded traffic sign.</p>", unsafe_allow_html=True)
st.markdown("---")

CLASS_NAMES = {
    0: "30km/h", 1: "50km/h", 2: "60km/h", 3: "70km/h",
    4: "Yield", 5: "Stop", 6: "No Entry",
    7: "General Caution", 8: "Road Work", 9: "Keep Right"
}

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('traffic_resnet_random.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Please ensure the model is uploaded and named correctly. Error: {e}")

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_column_width=True)
    
    image = image.convert('RGB')
    img_resized = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_processed = tf.keras.applications.resnet_v2.preprocess_input(img_array)
    
    predictions = model.predict(img_processed)
    predicted_class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    st.markdown("---")
    st.markdown("<h3>Analysis Complete</h3>", unsafe_allow_html=True)
    
    st.markdown(f"<h1 style='color: #A3A58E; font-size: 3rem;'>{CLASS_NAMES[predicted_class_idx]}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>Confidence Level: <b>{confidence:.1f}%</b></p>", unsafe_allow_html=True)