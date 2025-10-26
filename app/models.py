"""
MODEL LOADING MODULE
====================
This module is responsible for loading our three pre-trained AI models into memory.
Think of it as a "model loader" that brings our trained AI brains into the application.

Why separate loading from training?
- Training happens once (takes hours)
- Loading happens every time we start the app (takes seconds)
- Like the difference between baking a cake vs. taking it out of the fridge

The three models we load:
1. Detection Model (YOLO) - Finds traffic signs in images
2. Classification Model - Identifies the type of sign
3. Tampering Detector - Checks if signs are fake/modified

Each function tries multiple file paths because models might be saved in different locations.
"""

import os  # For file operations
import torch  # PyTorch deep learning framework
from pathlib import Path  # For handling file paths cleanly
from typing import Optional, List  # For type hints (documentation)

from utils.config_loader import ConfigLoader  # Loads configuration settings
from models import ModelFactory  # Factory pattern for creating models (used by tampered detector)


def load_detection_model() -> Optional[object]:
    """
    LOAD YOLO DETECTION MODEL
    ==========================
    This function loads the YOLO (You Only Look Once) model that finds traffic signs.
    
    YOLO is special because:
    - It's very fast (can process 30+ images per second)
    - It finds ALL signs in an image in one pass
    - It's widely used in self-driving cars and security systems
    
    The function checks three possible locations for the model file:
    1. runs/detect/traffic_sign_detection/weights/best.pt (standard training location)
    2. runs/detect/traffic_sign_detection/best.pt (alternative location)
    3. models/traffic_sign_detection.pt (final deployed location)
    
    Returns:
    - YOLO model object if found
    - None if not found (app will show a warning)
    """
    det_weights = Path('runs/detect/traffic_sign_detection/weights/best.pt')
    if det_weights.exists():
        from ultralytics import YOLO
        return YOLO(str(det_weights))
    fallback = Path('runs/detect/traffic_sign_detection/best.pt')
    if fallback.exists():
        from ultralytics import YOLO
        return YOLO(str(fallback))
    alt = Path('models/traffic_sign_detection.pt')
    if alt.exists():
        from ultralytics import YOLO
        return YOLO(str(alt))
    return None


# Classification model removed - YOLO already handles detection AND classification
# YOLO outputs both bounding boxes and class IDs for all 43 traffic sign types


def load_fake_detector():
    """Legacy function - redirects to load_tampered_detector for backward compatibility"""
    return load_tampered_detector()


def load_tampered_detector():
    """
    LOAD TAMPERED/FAKE SIGN DETECTOR MODEL
    =======================================
    This function loads the AI model that detects if a traffic sign has been tampered with.
    
    Why is this important?
    - Attackers can modify traffic signs to confuse self-driving cars
    - Examples: stickers on stop signs, graffiti, physical damage
    - This model acts as a "security guard" checking if signs are trustworthy
    
    How it works:
    - Binary classification: "Clean" vs "Tampered"
    - Trained on thousands of real and modified traffic signs
    - Looks for unusual patterns, stickers, or modifications
    
    Think of it as a counterfeit money detector, but for traffic signs!
    
    Returns:
    - Trained tampering detector model
    - None if model file not found
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # GPU or CPU
    cfg = ConfigLoader().config  # Load configuration
    
    # Try tampered_detection config first, fallback to fake_detection for backward compatibility
    detector_config = cfg.get('tampered_detection', cfg.get('fake_detection', {}))
    model_name = detector_config.get('model', 'resnet18')  # Default to ResNet18
    
    model = ModelFactory.create_fake_detector(model_name)  # Create model architecture

    # Try tampered detector checkpoints first, then fallback to fake detector
    ckpt_candidates: List[Path] = [
        Path('models/tampered_sign_detector.best.pth'),
        Path('models/tampered_sign_detector.pth'),
        Path('models/fake_sign_detector.best.pth'),
        Path('models/fake_sign_detector.pth')
    ]
    for ckpt in ckpt_candidates:
        if ckpt.exists():
            state = torch.load(str(ckpt), map_location=device)
            try:
                model.load_state_dict(state)
                model.to(device).eval()
                print(f"✅ Loaded tampered detector from: {ckpt}")
                return model
            except Exception as e:
                print(f"⚠️  Failed to load {ckpt}: {e}")
                pass
    
    print("⚠️  No tampered detector model found")
    return None


def get_class_names() -> List[str]:
    """
    GET HUMAN-READABLE NAMES FOR TRAFFIC SIGN CLASSES
    ==================================================
    This function loads the names of all 43 traffic sign classes in the correct order.
    
    Why do we need this?
    - AI models work with numbers (Class 0, Class 1, Class 2...)
    - Humans need readable names ("Speed Limit 20", "Stop", "Yield"...)
    - This function bridges the gap between machine and human understanding
    
    The function tries loading names from multiple sources (in priority order):
    1. config/class_names.txt - Plain text file, one name per line
    2. config/class_names.json - JSON array of names
    3. dataset YAML files - Names from training data configuration
    4. Fallback - Generic names like "Class_0", "Class_1", etc.
    
    IMPORTANT: The order MUST match the model's training order!
    - Class 0 in the file = Class 0 in the model
    - Mismatch = wrong predictions shown to users
    
    Think of it as a dictionary that translates "AI speak" to "human speak"
    
    Returns:
    - List of 43 class names in the correct order
    """
    # 1) Preferred method: plain text file, one name per line
    txt_path = Path('config/class_names.txt')
    if txt_path.exists():
        try:
            # Read file and remove empty lines
            lines = [ln.strip() for ln in txt_path.read_text(encoding='utf-8').splitlines() if ln.strip()]
            if lines:
                return lines
        except Exception:
            pass  # If file is corrupted, try next method

    # 2) JSON list fallback
    json_path = Path('config/class_names.json')
    if json_path.exists():
        try:
            import json
            data = json.loads(json_path.read_text(encoding='utf-8'))
            if isinstance(data, list) and data:
                return data
        except Exception:
            pass

    # 3) Use dataset YAML names if present
    import yaml
    yaml_paths = [
        Path('dataset/augmented/augmented_config.yaml'),
        Path('dataset/processed/traffic_sign_config.yaml')
    ]
    for p in yaml_paths:
        if p.exists():
            try:
                with open(p, 'r') as f:
                    cfg = yaml.safe_load(f)
                names = cfg.get('names') or cfg.get('classes')
                if isinstance(names, list) and names:
                    return names
            except Exception:
                continue

    # 4) Final fallback to generic classes (keep length to common 43)
    return [f'Class_{i}' for i in range(43)]


