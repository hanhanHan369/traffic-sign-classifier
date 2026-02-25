import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Vision | Traffic Classifier", layout="centered")

st.markdown("""
<style>
    /* Import Elegant Fonts from Google */
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Montserrat:wght@300;400;500&display=swap');

    /* Global Font Overrides */
    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
        color: #4A443B;
    }

    /* Elegant Headers */
    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #4A443B !important;
        text-align: center !important;
        font-weight: 400 !important;
        letter-spacing: 1px !important;
    }

    p {
        text-align: center;
        color: #6B655C !important;
    }

    /* Minimalist File Uploader Dropzone */
    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF !important;
        border: 1px dashed #A3A58E !important; 
        border-radius: 8px !important;
        padding: 2rem !important;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.02) !important;
    }

    /* Style the text inside the uploader */
    .st-emotion-cache-1gcek56 {
        color: #6B655C !important;
    }

    /* Elegant Button Styling */
    .stButton>button {
        border-radius: 25px !important;
        font-family: 'Montserrat', sans-serif !important;
        font-weight: 500 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        font-size: 0.85rem !important;
        border: 1px solid transparent !important;
        display: block !important;
        margin: 0 auto !important; 
    }
    
    /* Clean Image Display */
    [data-testid="stImage"] img {
        border-radius: 8px !important;
        box-shadow: 0px 8px 24px rgba(0,0,0,0.06) !important;
    }
    
            /* Add this into the <style> block of your 1_Classifier.py */
    @keyframes fadeUp {
        0% { opacity: 0; transform: translateY(20px); }
        100% { opacity: 1; transform: translateY(0); }
    }
    
    /* Animates the main container */
    .block-container {
        animation: fadeUp 0.8s ease forwards;
    }
    
    /* Animates the results dynamically */
    .stSuccess, h3 {
        animation: fadeUp 0.5s ease forwards;
    }
    
    /* Hide the default Streamlit top menu (Optional, but looks much cleaner) */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
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
# Make Prediction
    predictions = model.predict(img_array)[0] # Extract the 1D array of 10 probabilities
    
    # Get the indices of the top 3 highest probabilities
    top_3_indices = np.argsort(predictions)[::-1][:3]
    
    # Extract the names and confidences for the top 3
    top_3_classes = [CLASS_NAMES[i] for i in top_3_indices]
    top_3_confidences = predictions[top_3_indices] * 100
    
    st.markdown("---")
    st.markdown("<h3>Analysis Complete</h3>", unsafe_allow_html=True)
    
    # --- DISPLAY TOP 1 (The Winner) ---
    st.markdown(
        f"""
        <div style="text-align: center;">
            <h1 style='color: #A3A58E; font-size: 3.5rem; margin-bottom: 0px;'>{top_3_classes[0]}</h1>
            <p style='font-size: 1.2rem; color: #4A443B; margin-top: 5px;'>Confidence: <b>{top_3_confidences[0]:.1f}%</b></p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # --- DISPLAY #2 and #3 (Alternative Probabilities) ---
    # We use a subtle dividing line and softer colors to maintain the editorial aesthetic
    st.markdown("<hr style='border: 0.5px solid #D1CDC4; width: 40%; margin: 1.5rem auto;'>", unsafe_allow_html=True)
    
    st.markdown(
        f"""
        <div style="text-align: center; color: #6B655C;">
            <p style="font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;">Alternative Matches</p>
            <p style="margin: 4px; font-size: 1.05rem;"><b>2. {top_3_classes[1]}</b> &mdash; {top_3_confidences[1]:.1f}%</p>
            <p style="margin: 4px; font-size: 1.05rem;"><b>3. {top_3_classes[2]}</b> &mdash; {top_3_confidences[2]:.1f}%</p>
        </div>
        """, 
        unsafe_allow_html=True
    )