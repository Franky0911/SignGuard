"""
STREAMLIT WEB APPLICATION - MAIN INTERFACE
===========================================
This is the main web dashboard that users interact with to:
1. Upload traffic sign images
2. Take photos with their camera
3. Analyze traffic signs (detect, classify, check for tampering)
4. See AI explanations (XAI - Explainable AI)
5. View performance metrics
6. Download results

Think of this as the "front door" of the application - it's what users see and click on.
"""

# Import libraries for file handling, data manipulation, and visualization
import os  # For file system operations
from pathlib import Path  # For handling file paths in a clean way
import io  # For handling in-memory file operations
import base64  # For encoding images
import json  # For handling JSON data
import time  # For timing operations
import plotly.graph_objects as go  # For creating interactive charts
import plotly.express as px  # For quick visualizations
from plotly.subplots import make_subplots  # For creating chart layouts
import pandas as pd  # For data manipulation

# Streamlit is our web framework - it makes creating web apps easy
import streamlit as st
from PIL import Image  # For image processing
import numpy as np  # For numerical operations (arrays, math)
import cv2  # OpenCV - for computer vision tasks
import torch  # PyTorch - our deep learning framework

# Import our custom modules that handle the AI models
from app.models import load_detection_model, load_tampered_detector, get_class_names
from app.xai_utils import get_xai_explanations  # For explaining AI decisions
from app.tts_utils import synthesize_tts  # For text-to-speech audio generation

# Custom CSS for premium styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #1f77b4;
        --secondary-color: #ff7f0e;
        --success-color: #2ca02c;
        --danger-color: #d62728;
        --warning-color: #ff7f0e;
        --info-color: #17a2b8;
        --light-color: #f8f9fa;
        --dark-color: #343a40;
        --gradient-primary: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --gradient-secondary: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --border-radius: 12px;
    }

    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Header styling */
    .main-header {
        background: var(--gradient-primary);
        padding: 2rem;
        border-radius: var(--border-radius);
        margin-bottom: 2rem;
        box-shadow: var(--shadow);
        color: white;
        text-align: center;
    }

    .main-header h1 {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0;
    }

    /* Card styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 1rem;
    }

    .metric-card h3 {
        color: var(--primary-color);
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        font-weight: 600;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: var(--dark-color);
        margin-bottom: 0.5rem;
    }

    .metric-label {
        color: #6c757d;
        font-size: 0.9rem;
    }

    /* Upload area styling */
    .upload-area {
        border: 2px dashed var(--primary-color);
        border-radius: var(--border-radius);
        padding: 3rem 2rem;
        text-align: center;
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .upload-area:hover {
        border-color: var(--secondary-color);
        background: linear-gradient(135deg, #fff 0%, #f8f9fa 100%);
        transform: translateY(-2px);
        box-shadow: var(--shadow);
    }

    .upload-icon {
        font-size: 3rem;
        color: var(--primary-color);
        margin-bottom: 1rem;
    }

    /* Button styling */
    .stButton > button {
        background: var(--gradient-primary);
        color: white;
        border: none;
        border-radius: var(--border-radius);
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        box-shadow: var(--shadow);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    /* Camera section styling */
    .camera-section {
        background: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        margin-bottom: 2rem;
    }

    .camera-controls {
        display: flex;
        gap: 1rem;
        margin-bottom: 1rem;
        justify-content: center;
    }

    /* Performance metrics styling */
    .performance-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1rem;
        margin-bottom: 2rem;
    }

    /* XAI section styling */
    .xai-section {
        background: white;
        padding: 2rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        margin-bottom: 2rem;
    }

    .xai-tabs {
        border-bottom: 2px solid #e9ecef;
        margin-bottom: 2rem;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }

    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: var(--border-radius) var(--border-radius) 0 0;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        border: 1px solid #e9ecef;
    }

    .stTabs [aria-selected="true"] {
        background: var(--gradient-primary);
        color: white;
    }

    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background: var(--gradient-primary);
    }

    /* Alert styling */
    .stAlert {
        border-radius: var(--border-radius);
        border-left: 4px solid var(--info-color);
    }

    /* Image container styling */
    .image-container {
        background: white;
        padding: 1rem;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    /* Detection box styling */
    .detection-info {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: var(--border-radius);
        border-left: 4px solid var(--success-color);
        margin-bottom: 1rem;
    }

    /* Responsive design */
    @media (max-width: 768px) {
        .main-header h1 {
            font-size: 2rem;
        }
        
        .performance-grid {
            grid-template-columns: 1fr;
        }
        
        .camera-controls {
            flex-direction: column;
        }
    }
</style>
""", unsafe_allow_html=True)

st.set_page_config(
    page_title='SignGuard AI Dashboard', 
    layout='wide',
    initial_sidebar_state='expanded',
    page_icon='🚦'
)


def load_models_cached():
    """
    LOAD AI MODELS INTO MEMORY
    ===========================
    This function loads the two AI models we need:
    1. Detection model (YOLO) - finds traffic signs AND identifies their type (43 classes)
    2. Tampered detector - checks if the sign has been tampered with
    
    Note: YOLO handles both detection AND classification, so we don't need a separate classifier!
    
    The @st.cache_resource decorator means:
    - Models are loaded only ONCE when the app starts
    - They're kept in memory for fast access
    - No need to reload them for every user action
    
    Think of this like opening apps on your phone - they stay open in the background
    so you don't have to wait for them to start up each time.
    """
    @st.cache_resource(show_spinner=False)
    def _load():
        det = load_detection_model()  # YOLO model for finding signs AND classifying them
        tampered = load_tampered_detector()  # Model to check for tampering
        return det, tampered
    return _load()


def image_from_upload(uploaded) -> Path:
    """
    SAVE UPLOADED IMAGE TO TEMPORARY FOLDER
    ========================================
    When a user uploads an image through the web interface, we need to save it
    temporarily so our AI models can process it.
    
    Steps:
    1. Create a temporary folder called 'tmpUploads' if it doesn't exist
    2. Save the uploaded file there with its original name
    3. Return the path to the saved file
    
    Think of this like saving an email attachment to your Downloads folder.
    """
    tmp_dir = Path('tmpUploads')  # Define temporary storage location
    tmp_dir.mkdir(exist_ok=True)  # Create folder if it doesn't exist
    out_path = tmp_dir / uploaded.name  # Full path where file will be saved
    with open(out_path, 'wb') as f:  # Open file in write-binary mode
        f.write(uploaded.read())  # Write the uploaded data to file
    return out_path  # Return the path so we can use it later


def draw_detections(img: np.ndarray, detections: list) -> np.ndarray:
    """
    DRAW BOXES AROUND DETECTED TRAFFIC SIGNS
    =========================================
    This function takes an image and a list of detected signs, then draws:
    - Green rectangles around each sign
    - Text labels showing the class ID and confidence score
    
    Parameters:
    - img: The original image as a numpy array
    - detections: List of detected signs with bounding boxes
    
    Returns:
    - A copy of the image with boxes and labels drawn on it
    
    Think of this like circling important parts in a photo with a marker.
    """
    out = img.copy()  # Make a copy so we don't modify the original
    for det in detections:  # Loop through each detected sign
        x1, y1, x2, y2 = det['bbox']  # Get box coordinates
        conf = det.get('confidence', 0)  # Get confidence score (how sure AI is)
        cls = det.get('class_id', 0)  # Get class ID (type of sign)
        # Draw green rectangle around the sign
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add text label above the box
        cv2.putText(out, f"{cls}:{conf:.2f}", (x1, max(0, y1 - 5)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return out


def predict_pipeline(image_path: Path, det_model, tampered_model, class_names, conf_threshold=0.25):
    """
    MAIN AI PREDICTION PIPELINE
    ============================
    This is the CORE function that runs our entire traffic sign analysis:
    
    STEP 1: DETECTION & CLASSIFICATION - YOLO finds all traffic signs AND identifies their type
    STEP 2: TAMPERING CHECK - Determine if each sign has been tampered with/fake
    
    Parameters:
    - image_path: Path to the image file
    - det_model: YOLO detection model (handles both detection AND classification)
    - tampered_model: Tampering detection model
    - class_names: List of traffic sign class names
    - conf_threshold: Minimum confidence to accept a detection (default 0.25 = 25%)
    
    Returns:
    - Dictionary containing all results (detections, tampering predictions)
    
    Think of this as a 2-stage quality control process:
    1. "Find it and name it" (YOLO detection + classification)
    2. "Trust it?" (tampering check)
    """
    # Clear previous XAI results when starting new prediction
    for key in list(st.session_state.keys()):
        if key.startswith('xai_results_'):
            del st.session_state[key]
    
    # Load the image from disk
    img_bgr = cv2.imread(str(image_path))  # OpenCV loads images in BGR format
    if img_bgr is None:
        print(f"ERROR: Could not load image from {image_path}")
        return {}
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)  # Convert to RGB format
    
    # Debug image information
    print(f"DEBUG: Image loaded successfully")
    print(f"DEBUG: Image shape: {img_rgb.shape}")
    print(f"DEBUG: Image dtype: {img_rgb.dtype}")
    print(f"DEBUG: Image min/max values: {img_rgb.min()}/{img_rgb.max()}")

    results = {
        'image_path': str(image_path),
        'detections': [],  # YOLO detections with bbox, confidence, class_id, and class_name
        'tampered_predictions': []  # Tampering analysis for each detection
    }

    # Use improved detection if available
    try:
        from utils.improved_detection import create_improved_detector
        improved_detector = create_improved_detector()
        detections = improved_detector.detect(img_rgb)
        
        print(f"DEBUG: Improved detector found {len(detections)} detections")
        
        for detection in detections:
            results['detections'].append({
                'bbox': detection['bbox'],
                'confidence': detection['confidence'],
                'class_id': detection['class_id'],
                'class_name': detection.get('class_name', f"Class_{detection['class_id']}")
            })
            print(f"DEBUG: Added detection - Class: {detection['class_id']} ({detection.get('class_name', 'Unknown')}), Conf: {detection['confidence']:.3f}")
    
    except ImportError:
        print("DEBUG: Improved detection not available, falling back to standard detection")
        # Fallback to standard detection
        if det_model is not None:
            print(f"DEBUG: Detection model loaded successfully")
            print(f"DEBUG: Model type: {type(det_model)}")
            
            # Use configurable confidence threshold for initial detection
            yolo_results = det_model(img_rgb, conf=conf_threshold)
            
            # Debug: Print total detections found
            total_detections = 0
            if yolo_results and len(yolo_results) > 0 and hasattr(yolo_results[0], 'boxes') and yolo_results[0].boxes is not None:
                total_detections = len(yolo_results[0].boxes)
                print(f"DEBUG: Found {total_detections} detections with conf >= {conf_threshold}")
            
            for r in yolo_results:
                if getattr(r, 'boxes', None) is not None:
                    boxes = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    clss = r.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confs, clss):
                        # Additional confidence check (redundant but safe)
                        if conf >= conf_threshold:
                            x1, y1, x2, y2 = box.astype(int)
                            
                            # Validate bounding box dimensions
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Only add detections with reasonable size (avoid tiny detections)
                            if width >= 5 and height >= 5:  # Reduced from 10 to 5
                                class_id = int(cls)
                                class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                                results['detections'].append({
                                    'bbox': [x1, y1, x2, y2], 
                                    'confidence': float(conf), 
                                    'class_id': class_id,
                                    'class_name': class_name
                                })
                                print(f"DEBUG: Added detection - Class: {class_id} ({class_name}), Conf: {float(conf):.3f}, Size: {width}x{height}")
            
            # If no detections found with current threshold, try lower threshold for debugging
            if len(results['detections']) == 0:
                print(f"DEBUG: No detections with conf >= {conf_threshold}, trying lower threshold...")
                
                # Try multiple very low thresholds
                for test_threshold in [0.1, 0.05, 0.01, 0.005]:
                    print(f"DEBUG: Testing threshold {test_threshold}")
                    yolo_results_low = det_model(img_rgb, conf=test_threshold)
                    
                    for r in yolo_results_low:
                        if getattr(r, 'boxes', None) is not None:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            clss = r.boxes.cls.cpu().numpy()
                            
                            print(f"DEBUG: Threshold {test_threshold} found {len(boxes)} raw detections")
                            
                            if len(boxes) > 0:
                                print("DEBUG: Raw detection details:")
                                for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
                                    print(f"  Detection {i+1}: Class {int(cls)}, Conf {float(conf):.4f}, Box {box}")
                            
                            # Add ALL detections regardless of confidence for debugging
                            for box, conf, cls in zip(boxes, confs, clss):
                                x1, y1, x2, y2 = box.astype(int)
                                width = x2 - x1
                                height = y2 - y1
                                
                                # Very lenient size check for debugging
                                if width >= 1 and height >= 1:
                                    class_id = int(cls)
                                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                                    results['detections'].append({
                                        'bbox': [x1, y1, x2, y2], 
                                        'confidence': float(conf), 
                                        'class_id': class_id,
                                        'class_name': class_name
                                    })
                                    print(f"DEBUG: Added detection - Class: {class_id} ({class_name}), Conf: {float(conf):.4f}, Size: {width}x{height}")
                            
                            # If we found any detections, break out of threshold testing
                            if len(boxes) > 0:
                                break
                    
                    # If we found detections, break out of threshold loop
                    if len(results['detections']) > 0:
                        break
                
                # If still no detections, try with NO confidence threshold
                if len(results['detections']) == 0:
                    print("DEBUG: Still no detections, trying with NO confidence threshold...")
                    yolo_results_no_thresh = det_model(img_rgb, conf=0.0)
                    
                    for r in yolo_results_no_thresh:
                        if getattr(r, 'boxes', None) is not None:
                            boxes = r.boxes.xyxy.cpu().numpy()
                            confs = r.boxes.conf.cpu().numpy()
                            clss = r.boxes.cls.cpu().numpy()
                            
                            print(f"DEBUG: No threshold found {len(boxes)} detections")
                            if len(boxes) > 0:
                                print("DEBUG: All detections (no threshold):")
                                for i, (box, conf, cls) in enumerate(zip(boxes, confs, clss)):
                                    class_id = int(cls)
                                    class_name = class_names[class_id] if class_id < len(class_names) else f"Class_{class_id}"
                                    print(f"  Detection {i+1}: Class {class_id} ({class_name}), Conf {float(conf):.4f}")
                                    results['detections'].append({
                                        'bbox': box.astype(int).tolist(), 
                                        'confidence': float(conf), 
                                        'class_id': class_id,
                                        'class_name': class_name
                                    })

    # For each detection, run tampered detection if available
    from torchvision import transforms
    tfm = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for i, det in enumerate(results['detections']):
        x1, y1, x2, y2 = det['bbox']
        roi = img_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        # Tampered detection (Classification is already done by YOLO)
        if tampered_model is not None:
            tensor = tfm(roi).unsqueeze(0).to(next(tampered_model.parameters()).device)
            with torch.no_grad():
                logits = tampered_model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
                tampered_prob = float(probs[1].item())
                is_tampered = tampered_prob > 0.5
            results['tampered_predictions'].append({
                'detection_id': i,
                'label': 'Tampered' if is_tampered else 'Clean',
                'clean_probability': float(probs[0].item()),
                'tampered_probability': tampered_prob
            })

    return results


def section_home():
    st.markdown("""
    <div class="main-header">
        <h1>🚦 SignGuard AI Dashboard</h1>
        <p>Advanced Traffic Sign Recognition & Analysis Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature overview cards - properly aligned
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Detection & Classification</h3>
            <div class="metric-value">YOLO</div>
            <div class="metric-label">Detects and classifies 43 traffic sign types in real-time</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🛡️ Tampered Detection</h3>
            <div class="metric-value">ResNet18</div>
            <div class="metric-label">Identifies tampered and fake signs</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🧠 Explainable AI</h3>
            <div class="metric-value">XAI</div>
            <div class="metric-label">Grad-CAM++ & Integrated Gradients</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick start guide
    st.markdown("### 🚀 Quick Start Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **📸 Upload Image:**
        1. Go to the "Upload Image" tab
        2. Drag and drop your image or click to browse
        3. Click "Analyze" to process the image
        4. View detection results and explanations
        """)
    
    with col2:
        st.markdown("""
        **📹 Live Camera:**
        1. Go to the "Live Camera" tab
        2. Click "Activate Camera" to start
        3. Take a photo when ready
        4. Get instant analysis results
        """)
    
    # System status
    st.markdown("### 📊 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.success("✅ YOLO Model: Ready")
        st.caption("Handles both detection and classification")
    
    with col2:
        st.success("✅ Tampered Detection: Ready")
        st.caption("Identifies tampered signs")
    
    with col3:
        st.success("✅ XAI Module: Ready")
        st.caption("Explainable AI visualizations")


def section_upload(det, tampered, class_names, conf_threshold):
    st.markdown("### 📸 Image Upload & Analysis")
    
    # Premium upload area
    st.markdown("""
    <div class="upload-area">
        <div class="upload-icon">📁</div>
        <h3>Drag & Drop Your Image Here</h3>
        <p>or click to browse files</p>
        <small>Supported formats: PNG, JPG, JPEG (Max 10MB)</small>
    </div>
    """, unsafe_allow_html=True)
    
    upl = st.file_uploader('Upload Image', type=['png', 'jpg', 'jpeg'], label_visibility="collapsed")
    
    if not upl:
        st.info("👆 Please upload an image to get started")
        return
    
    # Show upload progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    status_text.text("📤 Uploading image...")
    progress_bar.progress(25)
    
    img_path = image_from_upload(upl)
    progress_bar.progress(50)
    status_text.text("🔄 Processing image...")
    
    # Display uploaded image
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        <div class="image-container">
            <h4>📷 Uploaded Image</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(str(img_path), caption='Input Image', width='stretch')
    
    with col2:
        st.markdown("""
        <div class="image-container">
            <h4>🔍 Analysis Results</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button('🚀 Analyze Image', width='stretch', type="primary"):
            with st.spinner('🔄 Analyzing image...'):
                progress_bar.progress(75)
                status_text.text("🧠 Running AI models...")
                
                results = predict_pipeline(img_path, det, tampered, class_names, conf_threshold)
                st.session_state['last_results'] = results
                
                progress_bar.progress(100)
                status_text.text("✅ Analysis complete!")
                
                # Show quick results summary
                if results['detections']:
                    st.success(f"🎯 Found {len(results['detections'])} traffic sign(s)")
                    
                    # Show detection details
                    with st.expander("🔍 Detection Details", expanded=False):
                        for i, det in enumerate(results['detections']):
                            st.write(f"**Detection {i+1}:**")
                            st.write(f"- Class ID: {det.get('class_id', 'Unknown')}")
                            det_class_name = det.get('class_name', f"Class_{det.get('class_id', 'Unknown')}")
                            st.write(f"- Detection Class: {det_class_name}")
                            st.write(f"- Confidence: {det.get('confidence', 0):.3f}")
                            st.write(f"- Bounding Box: {det['bbox']}")
                            st.write(f"- Size: {det['bbox'][2]-det['bbox'][0]}x{det['bbox'][3]-det['bbox'][1]} pixels")
                            st.write("---")
                    
                    # Show first detection summary
                    # Display detection class name (YOLO already classified it)
                    if results['detections']:
                        first_det = results['detections'][0]
                        det_class_name = first_det.get('class_name', f"Class_{first_det.get('class_id', 'Unknown')}")
                        st.info(f"🎯 Detection & Classification: **{det_class_name}** (Confidence: {first_det.get('confidence', 0):.2%})")
                    
                    if results['tampered_predictions']:
                        first_tampered = results['tampered_predictions'][0]
                        tampered_status = "🟢 Clean" if first_tampered['label'] == 'Clean' else "🔴 Tampered"
                        st.info(f"🛡️ Status: **{tampered_status}** (Confidence: {first_tampered['tampered_probability']:.2%})")
                else:
                    st.warning("⚠️ No traffic signs detected in the image")
                    st.info(f"💡 Try lowering the confidence threshold (currently {conf_threshold}) to detect more objects")
                    
                    # Additional debugging information
                    st.markdown("#### 🔍 Debug Information")
                    st.info("**Possible Issues:**")
                    st.write("1. **Class Mapping Mismatch**: Your model might be using different class IDs than expected")
                    st.write("2. **Image Quality**: The image might not be clear enough for detection")
                    st.write("3. **Model Training**: The model might not have been trained on similar images")
                    
                    st.markdown("**Debug Steps:**")
                    st.write("1. Check the console output for detailed detection information")
                    st.write("2. Try with a different image from your dataset")
                    st.write("3. Verify that Class 20 in your model actually represents speed limit 20")
                    
                    # Show class mapping for reference
                    with st.expander("📋 Current Class Mapping (First 25 classes)", expanded=False):
                        for i in range(min(25, len(class_names))):
                            st.write(f"Class {i}: {class_names[i]}")
                
                time.sleep(1)
                progress_bar.empty()
                status_text.empty()
                st.rerun()


def section_camera(det, tampered, class_names, conf_threshold):
    st.markdown("### 📹 Live Camera Analysis")
    
    st.markdown("""
    <div class="camera-section">
        <h4>🎥 Camera Controls</h4>
        <p>Click the button below to activate your camera and capture images for analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera activation button
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button('📷 Activate Camera', width='stretch', type="primary"):
            st.session_state['camera_active'] = True
            st.rerun()
    
    # Camera input only shows when activated
    if st.session_state.get('camera_active', False):
        st.markdown("""
        <div class="camera-section">
            <h4>📸 Camera Feed</h4>
            <p>Position your camera to capture traffic signs clearly, then click "Take Photo" below.</p>
        </div>
        """, unsafe_allow_html=True)
        
        cam = st.camera_input('Take a photo')
        
        if cam is not None:
            # Show processing animation
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            status_text.text("📤 Processing captured image...")
            progress_bar.progress(25)
            
            img = Image.open(cam)
            tmp = Path('tmpCamera')
            tmp.mkdir(exist_ok=True)
            p = tmp / 'capture.jpg'
            img.save(p)
            
            progress_bar.progress(50)
            status_text.text("🧠 Running AI analysis...")
            
            results = predict_pipeline(p, det, tampered, class_names, conf_threshold)
            st.session_state['last_results'] = results
            
            progress_bar.progress(100)
            status_text.text("✅ Analysis complete!")
            
            # Display results
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div class="image-container">
                    <h4>📷 Captured Image</h4>
                </div>
                """, unsafe_allow_html=True)
                st.image(str(p), caption='Captured Image', width='stretch')
            
            with col2:
                st.markdown("""
                <div class="image-container">
                    <h4>🔍 Analysis Results</h4>
                </div>
                """, unsafe_allow_html=True)
                
                if results['detections']:
                    st.success(f"🎯 Found {len(results['detections'])} traffic sign(s)")
                    
                    # Show detection summary
                    for i, det in enumerate(results['detections']):
                        with st.expander(f"Detection #{i+1}"):
                            st.write(f"**Detection Confidence:** {det.get('confidence', 0):.2%}")
                            st.write(f"**Bounding Box:** {det['bbox']}")
                            
                            # Detection class name (YOLO classification)
                            det_class_name = det.get('class_name', f"Class_{det.get('class_id', 'Unknown')}")
                            st.write(f"**🎯 Classified as:** {det_class_name}")
                            
                            # Tampered detection result
                            tampered_rows = [t for t in results['tampered_predictions'] if t['detection_id'] == i]
                            if tampered_rows:
                                trow = tampered_rows[0]
                                tampered_status = "🟢 Clean" if trow['label'] == 'Clean' else "🔴 Tampered"
                                st.write(f"**Status:** {tampered_status}")
                                st.write(f"**Clean Probability:** {trow['clean_probability']:.2%}")
                                st.write(f"**Tampered Probability:** {trow['tampered_probability']:.2%}")
                else:
                    st.warning("⚠️ No traffic signs detected in the captured image")
            
            time.sleep(1)
            progress_bar.empty()
            status_text.empty()
            
            # Deactivate camera after capture
            if st.button('🔄 Capture Another Photo', width='stretch'):
                st.session_state['camera_active'] = False
                st.rerun()
    
    else:
        st.info("👆 Click 'Activate Camera' above to start capturing images")


def section_prediction_xai(det, tampered, class_names, conf_threshold):
    st.markdown("### 🧠 Prediction & Explainable AI Analysis")
    
    results = st.session_state.get('last_results')
    if not results:
        st.info('👆 Please run a prediction from Upload or Live Camera first to see detailed analysis.')
        return

    # Display original image with detections
    st.markdown("#### 📷 Detection Results")
    
    img = cv2.cvtColor(cv2.imread(results['image_path']), cv2.COLOR_BGR2RGB)
    vis = draw_detections(img, results['detections']) if results['detections'] else img
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="image-container">
            <h4>🎯 Detected Traffic Signs</h4>
        </div>
        """, unsafe_allow_html=True)
        st.image(vis, caption='Image with Detections', width='stretch')
    
    with col2:
        st.markdown("""
        <div class="detection-info">
            <h4>📊 Detection Summary</h4>
        </div>
        """, unsafe_allow_html=True)
        
        if results['detections']:
            st.success(f"**Total Detections:** {len(results['detections'])}")
            
            # Show detection confidence distribution
            confidences = [det.get('confidence', 0) for det in results['detections']]
            avg_conf = sum(confidences) / len(confidences)
            st.metric("Average Confidence", f"{avg_conf:.2%}")
            
            # Classification is done by YOLO (detection confidence is also classification confidence)
            
            # Show tampered detection summary
            if results['tampered_predictions']:
                tampered_probs = [t['tampered_probability'] for t in results['tampered_predictions']]
                avg_tampered_prob = sum(tampered_probs) / len(tampered_probs)
                st.metric("Avg Tampered Probability", f"{avg_tampered_prob:.2%}")
        else:
            st.warning("No traffic signs detected")

    # Detailed analysis for each detection
    if results['detections']:
        st.markdown("#### 🔍 Detailed Analysis")
        
        for det_idx, detection in enumerate(results['detections']):
            with st.expander(f"🔍 Detection #{det_idx+1} - Detailed Analysis", expanded=False):
                
                # Detection info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("**📍 Bounding Box**")
                    st.write(f"X1: {detection['bbox'][0]}")
                    st.write(f"Y1: {detection['bbox'][1]}")
                    st.write(f"X2: {detection['bbox'][2]}")
                    st.write(f"Y2: {detection['bbox'][3]}")
                
                with col2:
                    st.markdown("**🎯 Detection Confidence**")
                    conf = detection.get('confidence', 0)
                    st.metric("Confidence", f"{conf:.2%}")
                    
                    # Confidence bar
                    st.progress(conf)
                
                with col3:
                    st.markdown("**📐 Dimensions**")
                    width = detection['bbox'][2] - detection['bbox'][0]
                    height = detection['bbox'][3] - detection['bbox'][1]
                    st.write(f"Width: {width}px")
                    st.write(f"Height: {height}px")
                    st.write(f"Area: {width * height}px²")
                
                # Show ROI
                x1, y1, x2, y2 = detection['bbox']
                roi = img[y1:y2, x1:x2]
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**🖼️ Region of Interest**")
                    st.image(roi, caption=f'ROI #{det_idx+1}', width='stretch')
                
                with col2:
                    st.markdown("**📋 Detection & Classification Results**")
                    
                    # Detection class name (YOLO already classified it)
                    det_class_name = detection.get('class_name', f"Class_{detection.get('class_id', 'Unknown')}")
                    st.success(f"**🎯 Classified as:** {det_class_name}")
                    st.metric("YOLO Confidence", f"{detection.get('confidence', 0):.2%}")
                    
                    # Confidence visualization
                    st.progress(detection.get('confidence', 0))
                    
                    st.info("ℹ️ YOLO handles both detection and classification in one pass")
                    
                    # Tampered detection results
                    tampered_rows = [t for t in results['tampered_predictions'] if t['detection_id'] == det_idx]
                    if tampered_rows:
                        trow = tampered_rows[0]
                        
                        st.markdown("**🛡️ Tampered Detection Analysis**")
                        
                        col_clean, col_tampered = st.columns(2)
                        
                        with col_clean:
                            clean_prob = trow['clean_probability']
                            st.metric("Clean Probability", f"{clean_prob:.2%}")
                            st.progress(clean_prob)
                        
                        with col_tampered:
                            tampered_prob = trow['tampered_probability']
                            st.metric("Tampered Probability", f"{tampered_prob:.2%}")
                            st.progress(tampered_prob)
                        
                        # Tampered detection verdict
                        verdict = "🟢 CLEAN" if trow['label'] == 'Clean' else "🔴 TAMPERED"
                        verdict_color = "success" if trow['label'] == 'Clean' else "error"
                        
                        if verdict_color == "success":
                            st.success(f"**Verdict:** {verdict}")
                        else:
                            st.error(f"**Verdict:** {verdict}")
                
                # XAI Analysis Section
                st.markdown("#### 🧠 Explainable AI Analysis")
                
                # Single XAI button for comprehensive analysis
                if st.button(f'🧠 Generate XAI Analysis for Detection #{det_idx+1}', key=f'xai_analysis_{det_idx}', width='stretch', type="primary"):
                    # Clear previous XAI results for this detection
                    if f'xai_results_{det_idx}' in st.session_state:
                        del st.session_state[f'xai_results_{det_idx}']
                    
                    # Save ROI to temp with unique timestamp
                    tmp = Path('tmpXAI')
                    tmp.mkdir(exist_ok=True)
                    
                    # Create unique filename with timestamp
                    timestamp = int(time.time() * 1000)  # milliseconds
                    roi_path = tmp / f'roi_{det_idx}_{timestamp}.png'
                    Image.fromarray(roi).save(roi_path)
                    
                    # Generate XAI explanations
                    with st.spinner(f"🔄 Generating XAI analysis for Detection #{det_idx+1}..."):
                        xai_results = {}
                        
                        # Tampered detection XAI
                        if tampered is not None:
                            try:
                                exp_tampered = get_xai_explanations(tampered, str(roi_path), 'tampered_detection')
                                xai_results['tampered_detection'] = exp_tampered
                            except Exception as e:
                                st.error(f"Tampered detection XAI failed: {str(e)}")
                                xai_results['tampered_detection'] = None

                        # Detection XAI (Grad-CAM++ and Integrated Gradients)
                        if det is not None:
                            try:
                                exp_det = get_xai_explanations(det, results['image_path'], 'detection')
                                xai_results['detection'] = exp_det
                            except Exception as e:
                                st.error(f"Detection XAI failed: {str(e)}")
                                xai_results['detection'] = None
                    
                    # Store results with unique key
                    st.session_state[f'xai_results_{det_idx}'] = {
                        'results': xai_results,
                        'timestamp': timestamp,
                        'roi_path': str(roi_path)
                    }
                    
                    st.success(f"✅ XAI analysis completed for Detection #{det_idx+1}")
                    st.rerun()
                
                # Display XAI results
                xai_data = st.session_state.get(f'xai_results_{det_idx}')
                if xai_data:
                    st.markdown("**🧠 XAI Analysis Results**")
                    
                    # Show analysis timestamp
                    st.info(f"Analysis generated at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(xai_data['timestamp']/1000))}")
                    
                    # Note: Classification XAI removed - YOLO's classification is part of detection
                    
                    # Tampered Detection XAI Results
                    if xai_data['results'].get('tampered_detection'):
                        exp_tampered = xai_data['results']['tampered_detection']
                        
                        st.markdown("**🛡️ Tampered Detection Analysis**")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Predicted:** {exp_tampered.get('predicted_class_name', 'N/A')}")
                            st.write(f"**Clean Probability:** {exp_tampered.get('clean_probability', 0):.2%}")
                            st.write(f"**Tampered Probability:** {exp_tampered.get('tampered_probability', 0):.2%}")
                            
                            # Show verdict
                            verdict = "🟢 CLEAN" if exp_tampered.get('tampered_probability', 0) < 0.5 else "🔴 TAMPERED"
                            if exp_tampered.get('tampered_probability', 0) < 0.5:
                                st.success(f"**Verdict:** {verdict}")
                            else:
                                st.error(f"**Verdict:** {verdict}")
                        
                        with col2:
                            st.markdown("**🖼️ Tampered Detection Visualizations**")
                            vis_cols = st.columns(2)
                            if exp_tampered.get('gradcam_pp_map') is not None:
                                vis_cols[0].image(exp_tampered['gradcam_pp_map'], caption='Grad-CAM++ (Spatial Attention)', width='stretch')
                            if exp_tampered.get('integrated_gradients') is not None:
                                vis_cols[1].image(exp_tampered['integrated_gradients'], caption='Integrated Gradients (Feature Attribution)', width='stretch')

                    # Detection XAI (YOLO) results
                    if xai_data['results'].get('detection'):
                        exp_det = xai_data['results']['detection']
                        st.markdown("**🎯 Detection Explainability (YOLO)**")
                        # Try to map to this detection index if arrays available
                        try:
                            gradcam_pp_maps = exp_det.get('gradcam_pp_maps', [])
                            ig_maps = exp_det.get('integrated_gradients_maps', [])
                            
                            cols = st.columns(2)
                            if len(gradcam_pp_maps) > 0 and gradcam_pp_maps[0] is not None:
                                cols[0].image(gradcam_pp_maps[0], caption='Grad-CAM++ (YOLO)', width='stretch')
                            else:
                                cols[0].info('Grad-CAM++ not available')
                            
                            if len(ig_maps) > 0 and ig_maps[0] is not None:
                                cols[1].image(ig_maps[0], caption='Integrated Gradients (YOLO)', width='stretch')
                            else:
                                cols[1].info('Integrated Gradients not available')
                        except Exception as e:
                            st.info(f'Detection XAI visualizations not available: {str(e)}')
                    
                    # Show ROI used for analysis
                    st.markdown("**📷 Region Analyzed**")
                    st.image(xai_data['roi_path'], caption=f'ROI used for XAI analysis - Detection #{det_idx+1}', width=200)
                
                else:
                    st.info("👆 Click 'Generate XAI Analysis' above to see detailed explanations for this detection")

    # Audio synthesis
    if results['detections']:
        st.markdown("#### 🔊 Audio Summary")
        
        first_det = results['detections'][0]
        first_tampered = results['tampered_predictions'][0] if results['tampered_predictions'] else None
        
        # Use YOLO's classification from detection
        det_class_name = first_det.get('class_name', f"Class_{first_det.get('class_id', 'Unknown')}")
        text = f"The detected sign is {det_class_name}."
        if first_tampered:
            text += f" It is classified as {first_tampered['label'].lower()}."
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.write(f"**Audio Text:** {text}")
        
        with col2:
            if st.button("🔊 Generate Audio", width='stretch'):
                audio = synthesize_tts(text)
                if audio:
                    st.audio(audio, format='audio/mp3')
                else:
                    st.info("Audio generation not available")


def section_downloads():
    st.header('Download Results')
    results = st.session_state.get('last_results')
    if not results:
        st.info('Run a prediction first.')
        return

    # CSV
    import csv
    csv_buf = io.StringIO()
    writer = csv.writer(csv_buf)
    writer.writerow(['image', 'x1', 'y1', 'x2', 'y2', 'yolo_conf', 'class_name', 'class_id', 'clean_prob', 'tampered_prob'])
    for i, det in enumerate(results['detections']):
        x1, y1, x2, y2 = det['bbox']
        tampered_rows = [t for t in results['tampered_predictions'] if t['detection_id'] == i]
        cls_name = det.get('class_name', f"Class_{det.get('class_id', 'Unknown')}")
        cls_id = det.get('class_id', '')
        clean_p = tampered_rows[0]['clean_probability'] if tampered_rows else ''
        tampered_p = tampered_rows[0]['tampered_probability'] if tampered_rows else ''
        writer.writerow([results['image_path'], x1, y1, x2, y2, det.get('confidence', ''), cls_name, cls_id, clean_p, tampered_p])
    st.download_button('Download CSV', csv_buf.getvalue(), file_name='results.csv', mime='text/csv')

    # PDF (simple fallback as text-only to avoid heavy deps)
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
        pdf_buf = io.BytesIO()
        c = canvas.Canvas(pdf_buf, pagesize=letter)
        textobject = c.beginText(40, 740)
        textobject.textLine(f"Image: {results['image_path']}")
        for i, det in enumerate(results['detections']):
            textobject.textLine(f"Det #{i+1}: bbox={det['bbox']} conf={det.get('confidence',0):.2f}")
            cls_name = det.get('class_name', f"Class_{det.get('class_id', 'Unknown')}")
            textobject.textLine(f"  Class: {cls_name} (YOLO confidence: {det.get('confidence',0):.2f})")
            tampered_rows = [t for t in results['tampered_predictions'] if t['detection_id'] == i]
            if tampered_rows:
                textobject.textLine(f"  Status: Clean {tampered_rows[0]['clean_probability']:.2f}, Tampered {tampered_rows[0]['tampered_probability']:.2f}")
        c.drawText(textobject)
        c.showPage(); c.save()
        st.download_button('Download PDF', pdf_buf.getvalue(), file_name='results.pdf', mime='application/pdf')
    except Exception:
        st.info('Install reportlab for PDF export.')


def main():
    # Sidebar navigation
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 1rem;">
        <h2>🚦 SignGuard AI</h2>
        <p>Traffic Sign Recognition Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "📋 Navigation",
        ["🏠 Home", "📸 Upload Image", "📹 Live Camera", "🧠 XAI Analysis", "📥 Downloads"]
    )
    
    # Sidebar info
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 System Status")
    
    # Load models with progress indicator
    with st.spinner("🔄 Loading AI models..."):
        det, tampered = load_models_cached()
        class_names = get_class_names()
    
    st.sidebar.success("✅ All models loaded successfully")
    
    # Model info
    st.sidebar.markdown("### 🤖 Model Information")
    st.sidebar.info(f"**Detection Model:** YOLO (with built-in classification)")
    st.sidebar.info(f"**Traffic Sign Classes:** {len(class_names)}")
    st.sidebar.info(f"**Tampered Detection Model:** ResNet18")
    
    # Detection settings
    st.sidebar.markdown("### ⚙️ Detection Settings")
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold", 
        min_value=0.01, 
        max_value=0.9, 
        value=0.15, 
        step=0.01,
        help="Lower values detect more objects but may include false positives. Recommended: 0.15 for unseen data"
    )
    st.sidebar.info(f"**Current Threshold:** {conf_threshold}")
    
    # Quick stats
    st.sidebar.markdown("### 📈 Quick Stats")
    if st.session_state.get('last_results'):
        results = st.session_state['last_results']
        if results['detections']:
            st.sidebar.metric("Last Analysis", f"{len(results['detections'])} signs detected")
        else:
            st.sidebar.metric("Last Analysis", "No signs detected")
    else:
        st.sidebar.metric("Last Analysis", "Not performed")
    
    # Main content based on selection
    if page == "🏠 Home":
        section_home()
    elif page == "📸 Upload Image":
        section_upload(det, tampered, class_names, conf_threshold)
    elif page == "📹 Live Camera":
        section_camera(det, tampered, class_names, conf_threshold)
    elif page == "🧠 XAI Analysis":
        section_prediction_xai(det, tampered, class_names, conf_threshold)
    elif page == "📥 Downloads":
        section_downloads()


if __name__ == '__main__':
    main()


