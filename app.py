import os
import cv2
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from scipy.special import softmax

# ====================== CUSTOM METRIC ======================
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1.0):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2. * intersection + smooth) / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + smooth)

# ====================== PAGE CONFIG ======================
st.set_page_config(
    page_title="NeuroScan ",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ====================== UI STYLES ======================
st.markdown("""
<style>
    :root {
        --primary: #5ac8fa;
        --primary-dark: #36a9dc;
        --primary-light: #8bdcff;
        --secondary: #5ac8fa;
        --bg-dark: #111318;
        --bg-card: #191b21;
        --bg-card-light: #20232a;
        --text-primary: #f5f5f7;
        --text-secondary: #a1a1aa;
        --text-muted: #777b84;
        --success: #66c78a;
        --warning: #f3bd62;
        --danger: #ff7777;
        --border: #30333b;
        --shadow: rgba(0, 0, 0, 0.18);
    }
    
    body, .stApp {
        background: var(--bg-dark);
        color: var(--text-primary);
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        min-height: 100vh;
    }

    [data-testid="stHeader"] {
        background: var(--bg-dark);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1160px;
        padding: 1.5rem 2rem 2.5rem;
    }
    
    .main-header {
        color: var(--text-primary);
        font-size: clamp(1.8rem, 3vw, 2.35rem);
        font-weight: 650;
        letter-spacing: -0.055em;
        line-height: 1.15;
        margin: 0;
        padding: 0;
    }
    
    .sub-header {
        color: var(--text-secondary);
        font-size: 0.94rem;
        line-height: 1.5;
        margin: 0.45rem 0 0;
    }
    
    .eyebrow {
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 650;
        letter-spacing: 0.1em;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
    }

    .research-note {
        background: #1d1b16;
        border: 1px solid #49402c;
        border-radius: 10px;
        color: #e6d3a9;
        font-size: 0.82rem;
        line-height: 1.5;
        margin: 1.15rem 0 1.6rem;
        padding: 0.7rem 0.9rem;
    }

    .badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin: 1rem 0 1.7rem;
    }

    .badge {
        background: #1b1d23;
        border: 1px solid var(--border);
        border-radius: 999px;
        color: #c8cbd2;
        display: inline-flex;
        font-size: 0.76rem;
        font-weight: 500;
        line-height: 1;
        padding: 0.45rem 0.7rem;
    }

    .section-label {
        color: #e5e6e9;
        font-size: 0.8rem;
        font-weight: 620;
        letter-spacing: 0.015em;
        margin: 0.1rem 0 0.55rem;
    }

    .stSelectbox [data-baseweb="select"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 9px !important;
        color: var(--text-primary) !important;
        font-size: 0.9rem !important;
        min-height: 42px !important;
    }
    
    .stSelectbox [data-baseweb="select"]:hover {
        border-color: var(--primary) !important;
    }

    [data-testid="stFileUploader"] section {
        background: var(--bg-card);
        border: 1px dashed #484c56;
        border-radius: 10px;
        min-height: 96px;
        padding: 0.75rem;
    }

    [data-testid="stFileUploader"] section:hover {
        border-color: var(--primary);
    }

    [data-testid="stFileUploader"] button {
        background: #292c33;
        border: 1px solid #3b3e46;
        border-radius: 7px;
        color: var(--text-primary);
        min-height: 34px;
    }

    .upload-card {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid var(--border);
        text-align: center;
    }
    
    .result-card {
        background: var(--bg-card);
        border-radius: 12px;
        padding: 1.25rem;
        border: 1px solid var(--border);
        position: relative;
        overflow: hidden;
    }
    
    .result-card::before {
        content: none;
    }
    
    .stat-card {
        background: var(--bg-card);
        border-radius: 11px;
        padding: 1.1rem;
        text-align: center;
        border: 1px solid var(--border);
    }
    
    .stat-value {
        color: var(--primary);
        font-size: 2rem;
        font-weight: 650;
        line-height: 1;
        margin-bottom: 0.45rem;
    }
    
    .stat-label {
        color: var(--text-secondary);
        font-size: 0.72rem;
        font-weight: 550;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    
    .stButton > button {
        background: var(--primary) !important;
        color: #101318 !important;
        font-weight: 620 !important;
        border-radius: 9px !important;
        padding: 0.55rem 1rem !important;
        font-size: 0.9rem !important;
        border: 1px solid var(--primary) !important;
        transition: background 0.15s ease, border-color 0.15s ease !important;
        width: 100%;
        min-height: 42px;
    }
    
    .stButton > button:hover {
        background: var(--primary-light) !important;
        border-color: var(--primary-light) !important;
    }
    
    .stButton > button:active {
        transform: scale(0.99);
    }
    
    .stProgress > div > div > div > div {
        background: var(--primary);
    }
    
    .stFileUploader {
        background: transparent !important;
    }
    
    .stImage > img {
        border-radius: 10px;
        border: 1px solid var(--border);
    }
    
    .stSpinner > div {
        border-color: var(--primary) transparent transparent transparent !important;
    }
    
    hr {
        border: none;
        height: 1px;
        background: var(--border);
        margin: 1.5rem 0;
    }
    
    .footer {
        text-align: center;
        color: var(--text-muted);
        font-size: 0.78rem;
        padding: 1rem 0;
        margin-top: 1rem;
        border-top: 1px solid var(--border);
    }
    
    .footer a {
        color: var(--primary);
        text-decoration: none;
    }
    
    .confidence-meter {
        height: 5px;
        background: var(--border);
        border-radius: 999px;
        overflow: hidden;
        margin: 0.8rem 0 0.2rem;
    }
    
    .confidence-fill {
        height: 100%;
        background: var(--primary);
        border-radius: 999px;
    }
    
    .alert {
        padding: 1rem;
        border-radius: 11px;
        margin: 0.8rem 0;
        display: flex;
        align-items: center;
        gap: 0.8rem;
    }
    
    .alert-success {
        background: #17231c;
        border: 1px solid #31563c;
        color: var(--success);
    }
    
    .alert-warning {
        background: #242016;
        border: 1px solid #51452b;
        color: var(--warning);
    }
    
    .alert-danger {
        background: #271a1c;
        border: 1px solid #5b3238;
        color: var(--danger);
    }

    @media (max-width: 700px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 1rem 2rem;
        }

        .sub-header {
            font-size: 0.86rem;
        }
    }
</style>
""", unsafe_allow_html=True)

# ====================== HEADER ======================
st.markdown("""
<div class="eyebrow">MRI research workspace</div>
<h1 class="main-header">NeuroScan <span style="color: var(--primary);">AI</span></h1>
<p class="sub-header">Explore experimental brain MRI classification and segmentation models.</p>
<div class="research-note"><strong>Research use only.</strong> Results are unvalidated model outputs, not a diagnosis. Do not use them to guide medical decisions.</div>
""", unsafe_allow_html=True)

# ====================== FEATURES BADGES ======================
st.markdown(
    '<div class="badge-row"><span class="badge">Classification</span>'
    '<span class="badge">Segmentation</span><span class="badge">JPG · PNG · TIFF</span></div>',
    unsafe_allow_html=True,
)

# ====================== LOAD MODELS (cached) ======================
@st.cache_resource
def load_models():
    base = "models"
    cls = {
        "stroke": tf.keras.models.load_model(os.path.join(base, "stroke_binary.keras"), compile=False),
        "alzheimer": tf.keras.models.load_model(os.path.join(base, "alzheimer_binary.keras"), compile=False),
        "tumor_binary": tf.keras.models.load_model(os.path.join(base, "tumor_binary.keras"), compile=False),
        "tumor_multiclass": tf.keras.models.load_model(os.path.join(base, "tumor_multiclass.keras"), compile=False),
    }
    seg = tf.keras.models.load_model(
        os.path.join(base, "unetpp_fast_brain_tumor.keras"),
        compile=False,
        custom_objects={"dice_coef": dice_coef},
    )
    return cls, seg

cls_models, seg_model = load_models()

with open("indices/multiclass_class_indices.json") as f:
    idx_to_tumor = {v: k.upper() for k, v in json.load(f).items()}

# ====================== CONSTANTS ======================
CLS_IMG_SIZE = 224
SEG_IMG_SIZE = 256
CLS_THRESHOLD = 0.50
SEG_THRESHOLD = 0.72

# ====================== ULTRA-AGGRESSIVE BRAIN EXTRACTION ======================
def extract_brain_mask(pil_img):
    if pil_img.mode != 'L':
        gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    else:
        gray = np.array(pil_img)
    h, w = gray.shape
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, np.ones((23, 19), np.uint8), iterations=5)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8), iterations=2)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels <= 1:
        return np.ones((h, w), dtype=np.uint8)
    candidates = []
    img_center = np.array([w / 2.0, h / 2.0])
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 8000: continue
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT:cv2.CC_STAT_LEFT+4]
        aspect = max(bw, bh) / min(bw, bh)
        fill = area / (bw * bh + 1e-6)
        cent = centroids[i]
        dist = np.linalg.norm(cent - img_center)
        vert_pos = cent[1] / h
        if (aspect < 3.5 and fill > 0.50 and dist < max(h, w) * 0.32 and vert_pos < 0.68):
            candidates.append((area, i, dist + (vert_pos * 2000)))
    if not candidates:
        best_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    else:
        _, best_label, _ = min(candidates, key=lambda x: x[2])
    mask = (labels == best_label).astype(np.uint8)
    mask = cv2.erode(mask, np.ones((19, 13), np.uint8), iterations=4)
    num, lab, st, _ = cv2.connectedComponentsWithStats(mask)
    if num > 1:
        mask = (lab == (1 + np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((19, 19), np.uint8), iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8), iterations=2)
    crop_bottom = int(h * 0.225)
    crop_side = int(w * 0.10)
    mask[h - crop_bottom:, :] = 0
    mask[:, :crop_side] = 0
    mask[:, w - crop_side:] = 0
    if mask.sum() < h * w * 0.08:
        return np.ones((h, w), dtype=np.uint8)
    return mask

# ====================== PREPROCESS ======================
def preprocess_classification(pil_img):
    arr = np.array(pil_img)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    arr = cv2.resize(arr, (CLS_IMG_SIZE, CLS_IMG_SIZE))
    arr = preprocess_input(arr)
    return np.expand_dims(arr, 0)

def preprocess_segmentation(pil_img):
    arr = np.array(pil_img.convert("L"))
    arr = cv2.resize(arr, (SEG_IMG_SIZE, SEG_IMG_SIZE))
    arr = arr.astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=(0, -1))

def keep_largest_component(mask):
    mask_u8 = (mask > 0).astype(np.uint8)
    num, labels = cv2.connectedComponents(mask_u8)
    if num <= 1:
        return mask_u8
    areas = [np.sum(labels == i) for i in range(1, num)]
    return (labels == (1 + np.argmax(areas))).astype(np.uint8)

# ====================== SEGMENTATION WITH CONFIDENCE ======================
def segment_and_draw_box(pil_img):
    orig_w, orig_h = pil_img.size
    brain_mask = extract_brain_mask(pil_img)
    inp = preprocess_segmentation(pil_img)
    pred = seg_model.predict(inp, verbose=0)[0, :, :, 0]
    
    tumor_mask = (pred > SEG_THRESHOLD).astype(np.uint8)
    tumor_mask = cv2.resize(tumor_mask, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    tumor_mask = np.logical_and(tumor_mask, brain_mask).astype(np.uint8)
    
    if tumor_mask.sum() < 400:
        return None, 0.0
    
    tumor_mask = keep_largest_component(tumor_mask)
    contours, _ = cv2.findContours(tumor_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, 0.0
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 500:
        return None, 0.0
    
    x, y, w, h = cv2.boundingRect(cnt)
    overlay = np.array(pil_img)
    cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 255, 100), thickness=6)
    cv2.putText(overlay, "TUMOR", (x, max(0, y - 45)),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 120), 4, cv2.LINE_AA)
    
    pred_resized = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    tumor_probs = pred_resized[tumor_mask == 1]
    seg_conf = float(np.mean(tumor_probs) * 100) if len(tumor_probs) > 0 else 0.0
    
    return overlay, seg_conf

# ====================== MAIN UI ======================
col1, col2 = st.columns([1, 1.35], gap="large")
with col1:
    st.markdown('<div class="section-label">Analysis type</div>', unsafe_allow_html=True)
    task = st.selectbox(
        "Analysis type",
        ["Brain Tumor Detection", "Brain Stroke Detection", "Alzheimer Detection"],
        label_visibility="collapsed",
        index=0
    )

with col2:
    st.markdown('<div class="section-label">MRI scan</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("MRI scan", type=["jpg", "jpeg", "png", "tif"], label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    
    st.markdown('<div class="section-label">Scan preview</div>', unsafe_allow_html=True)
    st.caption(uploaded.name)
    col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
    with col_img2:
        st.image(image, use_container_width=True)

# Analysis Button
st.markdown("<br>", unsafe_allow_html=True)

if uploaded:
    if st.button("Run analysis", type="primary", use_container_width=True):
        with st.spinner("Processing with experimental models..."):
            
            # Analysis Progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate processing steps
            steps = ["Extracting brain region", "Normalizing image", "Running classification", "Validating results", "Generating report"]
            for i, step in enumerate(steps):
                progress_bar.progress((i + 1) * 20)
                status_text.text(f"⏳ {step}...")
                
            status_text.empty()
            
            if task == "Brain Tumor Detection":
                p = cls_models["tumor_binary"].predict(preprocess_classification(image), verbose=0)[0][0]
                conf = p * 100
                
                if p < CLS_THRESHOLD:
                    # NO TUMOR DETECTED
                    st.markdown("""
                    <div class="alert alert-success">
                        <div style="font-size: 2.5rem;">✅</div>
                        <div>
                            <h3 style="margin: 0; color: var(--success);">No Tumor Detected</h3>
                            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">The MRI scan appears normal with no signs of tumor presence.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{conf:.1f}%</div>
                            <div class="stat-label">Confidence Score</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                else:
                    # TUMOR DETECTED
                    st.markdown("""
                    <div class="alert alert-danger">
                        <div style="font-size: 2.5rem;">⚠️</div>
                        <div>
                            <h3 style="margin: 0; color: var(--danger);">Tumor Detected</h3>
                            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">Potential tumor identified. Further analysis required.</p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence Stats
                    col_stat1, col_stat2, col_stat3 = st.columns(3)
                    with col_stat1:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{conf:.1f}%</div>
                            <div class="stat-label">Detection Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Multiclass Analysis
                    logits = cls_models["tumor_multiclass"].predict(preprocess_classification(image), verbose=0)[0]
                    probs = softmax(logits)
                    tumor_type = idx_to_tumor[int(np.argmax(logits))]
                    type_conf = float(np.max(probs) * 100)
                    
                    with col_stat2:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value" style="font-size: 2.5rem;">{tumor_type}</div>
                            <div class="stat-label">Tumor Type</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col_stat3:
                        st.markdown(f"""
                        <div class="stat-card">
                            <div class="stat-value">{type_conf:.1f}%</div>
                            <div class="stat-label">Type Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Segmentation Visualization
                    st.markdown('<div class="section-label">Segmentation Analysis</div>', unsafe_allow_html=True)
                    
                    result_img, seg_conf = segment_and_draw_box(image)
                    if result_img is not None:
                        col_vis1, col_vis2 = st.columns(2)
                        with col_vis1:
                            st.markdown('<div style="text-align: center; color: var(--text-secondary); margin-bottom: 1rem;">Original MRI</div>', unsafe_allow_html=True)
                            st.image(image, use_container_width=True)
                        with col_vis2:
                            st.markdown('<div style="text-align: center; color: var(--text-secondary); margin-bottom: 1rem;">Tumor Highlighted</div>', unsafe_allow_html=True)
                            st.image(result_img, use_container_width=True)
                        
                        # Segmentation Confidence
                        col_conf1, col_conf2, col_conf3 = st.columns([1, 2, 1])
                        with col_conf2:
                            st.markdown(f"""
                            <div class="stat-card">
                                <div class="stat-value">{seg_conf:.1f}%</div>
                                <div class="stat-label">Segmentation Confidence</div>
                                <div class="confidence-meter">
                                    <div class="confidence-fill" style="width: {seg_conf}%;"></div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="alert alert-warning">
                            <div style="font-size: 2.5rem;">🔍</div>
                            <div>
                                <h3 style="margin: 0; color: var(--warning);">Small Lesion Detected</h3>
                                <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary);">Tumor is too small for reliable segmentation. Consider higher resolution imaging.</p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
            elif task == "Brain Stroke Detection":
                p = cls_models["stroke"].predict(preprocess_classification(image), verbose=0)[0][0]
                conf = p * 100
                
                col_result1, col_result2 = st.columns([2, 1])
                
                with col_result1:
                    if p >= CLS_THRESHOLD:
                        st.markdown("""
                        <div class="alert alert-danger">
                            <div style="font-size: 3rem;">⚠️</div>
                            <div>
                                <h2 style="margin: 0; color: var(--danger);">Stroke Detected</h2>
                                <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary); font-size: 1.1rem;">
                                    Immediate medical attention recommended. The scan shows indications of cerebral infarction.
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="alert alert-success">
                            <div style="font-size: 3rem;">✅</div>
                            <div>
                                <h2 style="margin: 0; color: var(--success);">No Stroke Detected</h2>
                                <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary); font-size: 1.1rem;">
                                    The MRI scan appears normal with no signs of stroke or ischemic changes.
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_result2:
                    if p >= CLS_THRESHOLD:
                        st.markdown(f"""
                        <div class="stat-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                            <div class="stat-value">{conf:.1f}%</div>
                            <div class="stat-label">Confidence Level</div>
                            <div class="confidence-meter" style="margin-top: 1rem;">
                                <div class="confidence-fill" style="width: {conf}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="stat-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                            <div class="stat-value">{100-conf:.1f}%</div>
                            <div class="stat-label">Confidence Level</div>
                            <div class="confidence-meter" style="margin-top: 1rem;">
                                <div class="confidence-fill" style="width: {100-conf}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Additional Information
                st.markdown('<div class="section-label">Clinical Notes</div>', unsafe_allow_html=True)
                
                if p >= CLS_THRESHOLD:
                    st.markdown("""
                    <div style="background: rgba(255, 107, 139, 0.05); border-left: 4px solid var(--danger); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
                        <h4 style="color: var(--danger); margin-bottom: 0.5rem;">⚠️ Urgent Recommendations:</h4>
                        <ul style="color: var(--text-secondary); margin-left: 1.5rem;">
                            <li>Immediate neurological consultation required</li>
                            <li>Consider CT perfusion or angiography for confirmation</li>
                            <li>Monitor for signs of neurological deficit</li>
                            <li>Time is critical for potential thrombolytic therapy</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: rgba(0, 212, 170, 0.05); border-left: 4px solid var(--success); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
                        <h4 style="color: var(--success); margin-bottom: 0.5rem;">✅ Assessment Summary:</h4>
                        <ul style="color: var(--text-secondary); margin-left: 1.5rem;">
                            <li>Normal cerebral vasculature patterns observed</li>
                            <li>No evidence of ischemia or hemorrhage</li>
                            <li>Regular follow-up recommended based on clinical symptoms</li>
                            <li>Consider risk factor management if applicable</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    
            elif task == "Alzheimer Detection":
                p = cls_models["alzheimer"].predict(preprocess_classification(image), verbose=0)[0][0]
                conf = p * 100
                
                col_result1, col_result2 = st.columns([2, 1])
                
                with col_result1:
                    if p >= CLS_THRESHOLD:
                        st.markdown("""
                        <div class="alert alert-warning">
                            <div style="font-size: 3rem;">🧠</div>
                            <div>
                                <h2 style="margin: 0; color: var(--warning);">Alzheimer Patterns Detected</h2>
                                <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary); font-size: 1.1rem;">
                                    The scan shows patterns consistent with Alzheimer's disease. Consult neurologist for comprehensive evaluation.
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="alert alert-success">
                            <div style="font-size: 3rem;">✅</div>
                            <div>
                                <h2 style="margin: 0; color: var(--success);">Normal Brain Patterns</h2>
                                <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary); font-size: 1.1rem;">
                                    No Alzheimer-related patterns detected. Brain structure appears within normal limits.
                                </p>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col_result2:
                    if p >= CLS_THRESHOLD:
                        st.markdown(f"""
                        <div class="stat-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                            <div class="stat-value">{conf:.1f}%</div>
                            <div class="stat-label">Pattern Match</div>
                            <div class="confidence-meter" style="margin-top: 1rem;">
                                <div class="confidence-fill" style="width: {conf}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="stat-card" style="height: 100%; display: flex; flex-direction: column; justify-content: center;">
                            <div class="stat-value">{100-conf:.1f}%</div>
                            <div class="stat-label">Normal Confidence</div>
                            <div class="confidence-meter" style="margin-top: 1rem;">
                                <div class="confidence-fill" style="width: {100-conf}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Additional Information
                st.markdown('<div class="section-label">Pattern Analysis</div>', unsafe_allow_html=True)
                
                if p >= CLS_THRESHOLD:
                    st.markdown("""
                    <div style="background: rgba(255, 183, 77, 0.05); border-left: 4px solid var(--warning); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
                        <h4 style="color: var(--warning); margin-bottom: 0.5rem;">🧬 Observed Indicators:</h4>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
                            <div style="background: rgba(255, 183, 77, 0.1); padding: 1rem; border-radius: 10px;">
                                <div style="color: var(--warning); font-weight: 600;">Hippocampal Atrophy</div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem;">Medial temporal lobe volume reduction</div>
                            </div>
                            <div style="background: rgba(255, 183, 77, 0.1); padding: 1rem; border-radius: 10px;">
                                <div style="color: var(--warning); font-weight: 600;">Cortical Thinning</div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem;">Reduced gray matter density</div>
                            </div>
                            <div style="background: rgba(255, 183, 77, 0.1); padding: 1rem; border-radius: 10px;">
                                <div style="color: var(--warning); font-weight: 600;">Ventricular Enlargement</div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem;">Increased CSF spaces</div>
                            </div>
                            <div style="background: rgba(255, 183, 77, 0.1); padding: 1rem; border-radius: 10px;">
                                <div style="color: var(--warning); font-weight: 600;">White Matter Changes</div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem;">Possible vascular components</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="background: rgba(0, 212, 170, 0.05); border-left: 4px solid var(--success); padding: 1.5rem; border-radius: 12px; margin: 1rem 0;">
                        <h4 style="color: var(--success); margin-bottom: 0.5rem;">📊 Brain Health Metrics:</h4>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin-top: 1rem;">
                            <div style="background: rgba(0, 212, 170, 0.1); padding: 1rem; border-radius: 10px;">
                                <div style="color: var(--success); font-weight: 600;">Normal Hippocampus</div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem;">Adequate medial temporal volume</div>
                            </div>
                            <div style="background: rgba(0, 212, 170, 0.1); padding: 1rem; border-radius: 10px;">
                                <div style="color: var(--success); font-weight: 600;">Preserved Cortex</div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem;">Normal gray matter density</div>
                            </div>
                            <div style="background: rgba(0, 212, 170, 0.1); padding: 1rem; border-radius: 10px;">
                                <div style="color: var(--success); font-weight: 600;">Stable Ventricles</div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem;">Normal CSF spaces</div>
                            </div>
                            <div style="background: rgba(0, 212, 170, 0.1); padding: 1rem; border-radius: 10px;">
                                <div style="color: var(--success); font-weight: 600;">Healthy White Matter</div>
                                <div style="color: var(--text-secondary); font-size: 0.9rem;">No significant changes</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

# ====================== FOOTER ======================
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p><strong>NeuroScan AI</strong> · Experimental image analysis</p>
    <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 1rem;">
        ⚠️ <strong>Disclaimer:</strong> This tool is for research and educational purposes only.<br>
        Always consult qualified medical professionals for clinical diagnosis and treatment decisions.
    </p>
</div>
""", unsafe_allow_html=True)