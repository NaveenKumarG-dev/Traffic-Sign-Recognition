import os
import sys

# Add project root to path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import torch
from PIL import Image

from src.inference.predict import TrafficSignPredictor
from app.ui_components import (
    inject_custom_css,
    render_header,
    render_empty_state,
    render_prediction_card,
    render_top_predictions,
    render_model_info,
    render_pipeline
)

# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# Model Loading (cached)
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained model (cached - runs only once)."""
    try:
        predictor = TrafficSignPredictor()
        return predictor
    except FileNotFoundError:
        return None

# ============================================================
# Main Application
# ============================================================
def main():
    """Main Streamlit application."""
    
    # 1. Inject Theme & Layout Elements
    inject_custom_css()
    render_header()
    
    # 2. Load Model
    predictor = load_model()
    
    if predictor is None:
        st.error(
            "⚠️ **Model not found!** Please train the model first:\n\n"
            "```bash\npython main.py train\n```"
        )
        return
    
    # 3. Main Workspace Layout
    col1, col2 = st.columns([1, 1.2], gap="large")
    
    with col1:
        st.markdown('<div class="small-caps" style="margin-bottom: 0.5rem;">📷 INPUT IMAGE</div>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload a traffic sign image for AI classification",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.markdown('<div class="glass-card" style="margin-top: 1rem;">', unsafe_allow_html=True)
            st.image(image, use_container_width=True)
            st.markdown(f'<div class="muted-text" style="text-align: center; margin-top: 1rem;">{uploaded_file.name}</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        if uploaded_file is not None:
            with st.spinner("Analyzing traffic sign..."):
                # Prediction functionality unchanged
                result = predictor.predict(image)
            
            # Show prediction
            render_prediction_card(result)
            
            # Show top predictions
            render_top_predictions(result)
        else:
            render_empty_state()
            
    # 4. Footer & Technical Info
    st.markdown("<br><hr style='border-color: rgba(255,255,255,0.05);'>", unsafe_allow_html=True)
    render_model_info()
    render_pipeline()

if __name__ == "__main__":
    main()
