import streamlit as st

st.set_page_config(page_title="Vision | Home", page_icon="🌿", layout="wide")

# CSS with Advanced Animations and Card Designs
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=Montserrat:wght@300;400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Montserrat', sans-serif;
        color: #4A443B;
    }

    h1, h2, h3 {
        font-family: 'Playfair Display', serif !important;
        color: #4A443B !important;
        font-weight: 400 !important;
    }

    /* --- ANIMATIONS --- */
    @keyframes fadeUp {
        0% { opacity: 0; transform: translateY(30px); }
        100% { opacity: 1; transform: translateY(0); }
    }

    .animated-title {
        animation: fadeUp 1s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        text-align: center;
        font-size: 3.5rem !important;
        margin-bottom: 0px;
    }

    .animated-subtitle {
        animation: fadeUp 1s cubic-bezier(0.16, 1, 0.3, 1) 0.3s forwards;
        opacity: 0;
        text-align: center;
        color: #8C8E77 !important;
        font-size: 1.2rem;
        margin-bottom: 3rem;
    }

    /* --- INTRICATE CARDS --- */
    .info-card {
        background-color: #FFFFFF;
        border: 1px solid #E8EAE3;
        border-radius: 8px;
        padding: 2.5rem;
        height: 100%;
        box-shadow: 0px 10px 30px rgba(0, 0, 0, 0.02);
        animation: fadeUp 1s cubic-bezier(0.16, 1, 0.3, 1) 0.6s forwards;
        opacity: 0;
        transition: transform 0.3s ease;
    }
    
    .info-card:hover {
        transform: translateY(-5px);
    }

    .info-card h3 {
        color: #A3A58E !important;
        font-size: 1.5rem !important;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E8EAE3;
        padding-bottom: 0.5rem;
    }

    .info-card p {
        font-size: 0.95rem;
        line-height: 1.6;
        color: #6B655C;
    }

    /* Hide standard headers */
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# --- HERO SECTION ---
st.markdown("<h1 class='animated-title'>Vision by AI</h1>", unsafe_allow_html=True)
st.markdown("<p class='animated-subtitle'>Smart Traffic Sign Classification for Advanced Driver Assistance Systems.</p>", unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)

# --- GRID LAYOUT SECTION ---
col1, col2, col3 = st.columns([1, 10, 1])

with col2:
    card_col1, card_col2 = st.columns(2)
    
    with card_col1:
        st.markdown("""
        <div class="info-card">
            <h3>The Challenge</h3>
            <p>Autonomous systems and ADAS demand near-instant, highly accurate traffic sign recognition. However, these systems do not operate in a vacuum. They must reliably read signs through rain, snow, motion blur, and poor lighting conditions without sacrificing latency.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with card_col2:
        st.markdown("""
        <div class="info-card">
            <h3>The Architecture</h3>
            <p>To solve this, Vision employs a heavily optimized ResNet50V2 Transfer Learning architecture. By utilizing deep pre-trained visual filters and a custom fine-tuned classification head, the model instantly categorizes 10 distinct geometric street signs under real-world conditions.</p>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #8C8E77; animation: fadeUp 1s ease 1s forwards; opacity: 0;'>👈 Use the sidebar menu to launch the Classifier.</p>", unsafe_allow_html=True)