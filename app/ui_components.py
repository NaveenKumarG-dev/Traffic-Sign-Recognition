import streamlit as st
import torch
from src.config import GTSRB_CLASSES

def inject_custom_css():
    st.markdown("""
    <style>
        /* Base Theme */
        [data-testid="stAppViewContainer"] {
            background-color: #0B1120;
            color: #F8FAFC;
        }
        [data-testid="stHeader"] {
            background-color: transparent;
        }
        
        /* Typography */
        h1, h2, h3, h4, h5, h6, p, span, div {
            font-family: 'Inter', -apple-system, sans-serif;
            color: #F8FAFC;
        }
        .muted-text {
            color: #94A3B8 !important;
            font-size: 0.85rem;
        }
        .small-caps {
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94A3B8;
            font-weight: 600;
        }
        
        /* Header Hero */
        .hero-container {
            border-bottom: 1px solid #172033;
            padding-bottom: 1.5rem;
            margin-bottom: 2rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .hero-title {
            font-size: 2rem;
            font-weight: 700;
            margin: 0;
            display: flex;
            align-items: center;
            gap: 0.75rem;
        }
        .hero-subtitle {
            color: #94A3B8;
            font-size: 1rem;
            margin-top: 0.25rem;
        }
        .status-badge {
            background-color: rgba(34, 197, 94, 0.1);
            color: #22C55E;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            border: 1px solid rgba(34, 197, 94, 0.2);
        }
        .cuda-badge {
            background-color: rgba(56, 189, 248, 0.1);
            color: #38BDF8;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            border: 1px solid rgba(56, 189, 248, 0.2);
            margin-top: 0.5rem;
        }

        /* Cards */
        .glass-card {
            background-color: #172033;
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.5rem;
        }
        
        /* Prediction Area */
        .pred-title {
            font-size: 2.5rem;
            font-weight: 800;
            color: #38BDF8;
            line-height: 1.2;
            margin: 0.5rem 0;
            text-transform: uppercase;
        }
        .pred-conf {
            font-size: 1.5rem;
            font-weight: 600;
            margin-bottom: 0.5rem;
        }
        .conf-high { color: #22C55E; }
        .conf-good { color: #38BDF8; }
        .conf-mod { color: #F59E0B; }
        .conf-low { color: #EF4444; }
        
        /* Progress bars */
        .stProgress > div > div > div > div {
            background-color: #38BDF8;
        }
        
        /* Uploader override */
        [data-testid="stFileUploadDropzone"] {
            background-color: #172033 !important;
            border: 1px dashed rgba(255, 255, 255, 0.2) !important;
            border-radius: 12px !important;
        }
        [data-testid="stFileUploadDropzone"]:hover {
            border-color: #38BDF8 !important;
        }
        
        /* Empty State */
        .empty-state {
            text-align: center;
            padding: 4rem 2rem;
            background-color: #172033;
            border-radius: 12px;
            border: 1px dashed rgba(255, 255, 255, 0.1);
        }
        
        /* Top Preds */
        .top-pred-row {
            display: flex;
            align-items: center;
            margin-bottom: 0.75rem;
            font-size: 0.9rem;
        }
        .pred-rank {
            color: #94A3B8;
            font-weight: 600;
            width: 30px;
        }
        .pred-name {
            flex-grow: 1;
            padding-right: 1rem;
        }
        .pred-value {
            font-weight: 600;
            width: 60px;
            text-align: right;
        }
        .pred-bar-bg {
            width: 100%;
            height: 6px;
            background-color: rgba(255,255,255,0.1);
            border-radius: 3px;
            margin-top: 4px;
            overflow: hidden;
        }
        .pred-bar-fill {
            height: 100%;
            background-color: #38BDF8;
            border-radius: 3px;
        }
    </style>
    """, unsafe_allow_html=True)

def render_header():
    has_cuda = torch.cuda.is_available()
    cuda_badge = '<div class="cuda-badge">CUDA ACCELERATED</div>' if has_cuda else ''
    
    st.markdown(f"""
    <div class="hero-container">
        <div>
            <h1 class="hero-title">🚦 TRAFFIC SIGN RECOGNITION</h1>
            <div class="hero-subtitle">AI-Powered Driver Assistance System</div>
            <div class="muted-text" style="margin-top: 0.5rem;">GTSRB • PyTorch CNN • 43 Traffic Sign Classes</div>
            {cuda_badge}
        </div>
        <div style="text-align: right;">
            <div class="status-badge">● MODEL ONLINE</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_empty_state():
    st.markdown("""
    <div class="empty-state">
        <div style="font-size: 3rem; margin-bottom: 1rem;">🚦</div>
        <h3 style="margin-bottom: 0.5rem;">READY FOR ANALYSIS</h3>
        <p class="muted-text">Upload a traffic sign image<br>to begin classification.</p>
        <div class="small-caps" style="margin-top: 2rem;">Supported formats: PNG • JPG • JPEG</div>
    </div>
    """, unsafe_allow_html=True)

def render_prediction_card(result):
    conf = result["confidence"]
    name = result["class_name"]
    
    if conf > 90:
        status_color = "conf-high"
        status_text = "✓ HIGH CONFIDENCE"
        bar_color = "#22C55E"
    elif conf > 70:
        status_color = "conf-good"
        status_text = "✓ GOOD CONFIDENCE"
        bar_color = "#38BDF8"
    elif conf > 50:
        status_color = "conf-mod"
        status_text = "⚠ MODERATE CONFIDENCE"
        bar_color = "#F59E0B"
    else:
        status_color = "conf-low"
        status_text = "⚠ LOW CONFIDENCE"
        bar_color = "#EF4444"
        
    st.markdown(f"""<div class="glass-card">
<div class="small-caps" style="margin-bottom: 1rem;">🎯 AI PREDICTION</div>
<div class="pred-title">{name}</div>
<div style="margin-top: 2rem;">
<div style="display: flex; justify-content: space-between; align-items: flex-end;">
<div class="pred-conf">{conf:.2f}%</div>
</div>
<div class="pred-bar-bg" style="height: 8px; margin-bottom: 0.5rem;">
<div class="pred-bar-fill" style="width: {conf}%; background-color: {bar_color};"></div>
</div>
<div class="{status_color}" style="font-size: 0.85rem; font-weight: 600;">{status_text}</div>
</div>
</div>
<div class="status-badge" style="margin-top: 1rem; margin-bottom: 1rem;">✓ ANALYSIS COMPLETE</div>
<span class="muted-text" style="margin-left: 0.5rem;">Inference completed on {"CUDA" if torch.cuda.is_available() else "CPU"}</span>""", unsafe_allow_html=True)
    
    st.markdown(f"""<div class="glass-card" style="margin-top: 1rem;">
<div class="small-caps" style="margin-bottom: 0.5rem;">AI ANALYSIS</div>
<p class="muted-text" style="margin: 0; line-height: 1.5;">
The model classified the uploaded image as <strong>{name}</strong> 
with a confidence of <strong>{conf:.2f}%</strong>. 
The prediction was generated using a CNN trained on the 
German Traffic Sign Recognition Benchmark (GTSRB).
</p>
</div>""", unsafe_allow_html=True)

def render_top_predictions(result):
    st.markdown('<div class="small-caps" style="margin: 2rem 0 1rem 0;">TOP PREDICTIONS</div>', unsafe_allow_html=True)
    
    html = '<div class="glass-card">\n'
    for i, pred in enumerate(result["top_predictions"][:5], 1):
        conf = pred["confidence"]
        name = pred["class_name"]
        
        html += f'<div class="top-pred-row">\n'
        html += f'    <div class="pred-rank">{(i):02d}</div>\n'
        html += f'    <div class="pred-name">{name}\n'
        html += f'        <div class="pred-bar-bg">\n'
        html += f'            <div class="pred-bar-fill" style="width: {conf}%;"></div>\n'
        html += f'        </div>\n'
        html += f'    </div>\n'
        html += f'    <div class="pred-value">{conf:.2f}%</div>\n'
        html += f'</div>\n'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_model_info():
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
    with st.expander("▸ About this model"):
        st.markdown(f"""<div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
<div>
<div class="small-caps">Architecture</div>
<div>CNN</div>
</div>
<div>
<div class="small-caps">Dataset</div>
<div>German Traffic Sign Recognition Benchmark</div>
</div>
<div>
<div class="small-caps">Classes</div>
<div>43</div>
</div>
<div>
<div class="small-caps">Input Resolution</div>
<div>32 × 32</div>
</div>
<div>
<div class="small-caps">Framework</div>
<div>PyTorch</div>
</div>
<div>
<div class="small-caps">Acceleration</div>
<div>CUDA</div>
</div>
<div style="grid-column: span 2;">
<div class="small-caps">GPU</div>
<div>{gpu_name}</div>
</div>
</div>""", unsafe_allow_html=True)

def render_pipeline():
    with st.expander("▸ How the AI works"):
        st.markdown("""<div style="text-align: center; font-family: monospace; color: #94A3B8; line-height: 1.8;">
IMAGE<br>
↓<br>
RESIZE & NORMALIZE<br>
↓<br>
CNN FEATURE EXTRACTION<br>
↓<br>
43-CLASS CLASSIFICATION<br>
↓<br>
SOFTMAX PROBABILITIES<br>
↓<br>
PREDICTION
</div>""", unsafe_allow_html=True)
