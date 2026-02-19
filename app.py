import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration (Centered layout fits the aesthetic better)
st.set_page_config(page_title="Vision | AI Traffic Classifier", page_icon="🌿", layout="centered")

# 2. Injecting Custom "Aesthetic" CSS
st.markdown("""
<style>
    /* Import Elegant Fonts from Google */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Montserrat:wght@300;400;500&display=swap');

    /* Main Background and Text Settings */
    .stApp {
        background-color: #F4F2EB; /* Soft warm beige/cream */
        color: #4A443B; /* Deep earthy brown/grey for softer contrast than black */
        font-family: 'Montserrat', sans-serif;
    }

    /* Header Styling */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif;
        color: #4A443B;
        text-align: center;
        font-weight: 400;
        letter-spacing: 1px;
    }

    /* Subtitle/Text Styling */
    p {
        font-family: 'Montserrat', sans-serif;
        text-align: center;
        color: #6B655C;
    }

    /* File Uploader Styling */
    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF;
        border: 1px solid #D1CDC4;
        border-radius: 2px; /* Slight rounding, keeps it modern */
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.03);
    }

    /* Button Styling */
    .stButton>button {
        background-color: #A3A58E; /* Sage green from your image */
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
        margin: 0 auto; /* Centers the button */
    }
    
    .stButton>button:hover {
        background-color: #8C8E77; /* Darker sage on hover */
        color: white;
    }
    
    /* Image Display Styling */
    [data-testid="stImage"] img {
        border-radius: 4px;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.08);
    }
    
    /* Success/Error Message Styling */
    .stSuccess {
        background-color: #E8EAE3 !important;
        color: #4A443B !important;
        border: none !important;
    }
</style>
""", unsafe_allow_html=True)

# 3. App Content
st.markdown("<h1>Vision by AI</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload a roadside capture, and our ResNet architecture will seamlessly classify the traffic sign.</p>", unsafe_allow_html=True)
st.markdown("---")

CLASS_NAMES = {
    0: "30km/h", 1: "50km/h", 2: "60km/h", 3: "70km/h",
    4: "Yield", 5: "Stop", 6: "No Entry",
    7: "General Caution", 8: "Road Work", 9: "Keep Right"
}

# 4. Load the Model 
@st.cache_resource
def load_model():
    # Make sure this matches your newly saved .keras file!
    return tf.keras.models.load_model('traffic_resnet_final.keras')

try:
    model = load_model()
except Exception as e:
    st.error(f"Please ensure the model is uploaded and named correctly. Error: {e}")

# 5. File Uploader UI
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    # Create columns to center the image nicely
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_column_width=True)
    
    # Preprocess the image
    image = image.convert('RGB')
    img_resized = image.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_processed = tf.keras.applications.resnet_v2.preprocess_input(img_array)
    
    # Make Prediction
    predictions = model.predict(img_processed)
    predicted_class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    st.markdown("---")
    st.markdown("<h3>Analysis Complete</h3>", unsafe_allow_html=True)
    
    # Elegant results display
    st.markdown(f"<h1 style='color: #A3A58E; font-size: 3rem;'>{CLASS_NAMES[predicted_class_idx]}</h1>", unsafe_allow_html=True)
    st.markdown(f"<p>Confidence Level: <b>{confidence:.1f}%</b></p>", unsafe_allow_html=True)