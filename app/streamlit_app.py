"""
Traffic Sign Recognition System - Streamlit Application
========================================================
GTSRB-based PyTorch CNN Classifier

Upload a traffic sign image and get instant classification with confidence scores.
"""

import os
import sys

# Add project root to path
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(APP_DIR)
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import torch
from PIL import Image

from src.config import GTSRB_CLASSES, load_config
from src.inference.predict import TrafficSignPredictor


# ============================================================
# Page Configuration
# ============================================================
st.set_page_config(
    page_title="Traffic Sign Recognition",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Custom CSS
# ============================================================
st.markdown("""
<style>
    /* Global Background */
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #1e1e2f 0%, #2a2a40 100%);
        color: #e0e0e0;
    }
    
    /* Sidebar Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(30, 30, 47, 0.6) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Typography */
    h1, h2, h3, h4, h5, h6, p, span {
        color: #ffffff !important;
        font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
    }

    /* Prediction Box Glassmorphism */
    .prediction-box {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(16px) saturate(180%);
        -webkit-backdrop-filter: blur(16px) saturate(180%);
        border: 1px solid rgba(255, 255, 255, 0.12);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        text-align: center;
        margin: 1rem 0;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
        transition: transform 0.3s ease;
    }
    
    .prediction-box:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(255, 255, 255, 0.3);
    }

    .prediction-box h2 {
        background: -webkit-linear-gradient(45deg, #00d2ff, #3a7bd5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
        font-weight: 800;
    }
    
    .prediction-box h1 {
        font-size: 3.5rem;
        margin: 1rem 0;
        text-shadow: 0 0 20px rgba(0, 210, 255, 0.5);
    }
    
    /* File Uploader override */
    [data-testid="stFileUploadDropzone"] {
        background: rgba(255, 255, 255, 0.02) !important;
        backdrop-filter: blur(10px);
        border: 2px dashed rgba(255, 255, 255, 0.2) !important;
        border-radius: 16px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploadDropzone"]:hover {
        background: rgba(255, 255, 255, 0.08) !important;
        border: 2px dashed rgba(0, 210, 255, 0.6) !important;
    }

    /* Cards / Containers */
    div.stSpinner > div {
        border-color: #00d2ff transparent transparent transparent !important;
    }
    
    /* Progress Bars styling */
    .stProgress > div > div > div > div {
        background-image: linear-gradient(to right, #00d2ff, #3a7bd5);
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Model Loading (cached)
# ============================================================
@st.cache_resource
def load_model():
    """Load the trained model (cached - runs only once)."""
    try:
        predictor = TrafficSignPredictor()
        return predictor
    except FileNotFoundError as e:
        return None


# ============================================================
# Sidebar
# ============================================================
def render_sidebar():
    """Render the sidebar with model information."""
    st.sidebar.title("🔧 Model Information")
    
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    
    st.sidebar.markdown("---")
    
    info_items = {
        "Framework": "PyTorch",
        "Dataset": "GTSRB",
        "Classes": "43",
        "Input Size": "32 × 32",
        "Device": device,
        "GPU": gpu_name if torch.cuda.is_available() else "N/A",
    }
    
    for key, value in info_items.items():
        st.sidebar.markdown(f"**{key}:** {value}")
    
    st.sidebar.markdown("---")
    st.sidebar.title("📋 How it Works")
    st.sidebar.markdown("""
    ```
    Upload Image
          ↓
    Preprocessing
          ↓
    CNN Model
          ↓
    43-Class Prediction
          ↓
    Traffic Sign + Confidence
    ```
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.title("📊 Supported Signs")
    st.sidebar.markdown(f"The model recognizes **{len(GTSRB_CLASSES)}** different traffic sign types:")
    
    with st.sidebar.expander("View all classes"):
        for class_id, class_name in GTSRB_CLASSES.items():
            st.sidebar.markdown(f"`{class_id:02d}` {class_name}")


# ============================================================
# Main Application
# ============================================================
def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">', unsafe_allow_html=True)
    st.title("🚦 Traffic Sign Recognition System")
    st.markdown("**GTSRB-based PyTorch CNN Classifier**")
    st.markdown("Upload a traffic sign image to get instant classification")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Render sidebar
    render_sidebar()
    
    # Load model
    predictor = load_model()
    
    if predictor is None:
        st.error(
            "⚠️ **Model not found!** Please train the model first:\n\n"
            "```bash\npython main.py train\n```"
        )
        return
    
    st.success(f"✅ Model loaded on **{'CUDA' if torch.cuda.is_available() else 'CPU'}**")
    
    # File uploader
    st.markdown("---")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a traffic sign image",
            type=["png", "jpg", "jpeg"],
            help="Upload a PNG, JPG, or JPEG image of a traffic sign",
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        if uploaded_file is not None:
            st.subheader("🎯 Prediction")
            
            with st.spinner("Analyzing traffic sign..."):
                result = predictor.predict(image)
            
            # Main prediction
            st.markdown(
                f"""
                <div class="prediction-box">
                    <h2>🚗 {result['class_name']}</h2>
                    <p style="font-size: 1.2em; opacity: 0.9;">Class ID: {result['class_id']}</p>
                    <h1 style="color: white;">{result['confidence']:.2f}%</h1>
                    <p style="opacity: 0.7;">Confidence</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            
            # Top predictions
            st.subheader("📊 Top Predictions")
            
            for i, pred in enumerate(result["top_predictions"][:5], 1):
                conf = pred["confidence"]
                
                col_rank, col_name, col_bar = st.columns([0.5, 3, 2])
                
                with col_rank:
                    st.markdown(f"**#{i}**")
                
                with col_name:
                    st.markdown(f"{pred['class_name']}")
                
                with col_bar:
                    st.progress(conf / 100.0, text=f"{conf:.2f}%")
        else:
            st.subheader("🎯 Prediction")
            st.info("👆 Upload an image to get started!")
            
            # Show some example classes
            st.markdown("### Example Traffic Signs")
            st.markdown("The model can recognize signs such as:")
            
            example_classes = [2, 14, 13, 25, 38, 35, 1, 17]
            cols = st.columns(4)
            for idx, class_id in enumerate(example_classes):
                with cols[idx % 4]:
                    st.markdown(f"**{GTSRB_CLASSES[class_id]}**")


if __name__ == "__main__":
    main()
