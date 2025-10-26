"""
MODEL ARCHITECTURES AND FACTORY
================================
This module defines the structure (architecture) of our three AI models.

What is a model architecture?
- Like a blueprint for a building
- Defines the layers, connections, and structure of the neural network
- Each architecture is designed for a specific task

Our Three Model Architectures:

1. TrafficSignDetector (YOLO-based)
   - For FINDING traffic signs in images
   - Uses YOLO (You Only Look Once) - very fast detection
   - Can process 30+ images per second
   
2. TrafficSignClassifier (EfficientNet-based)
   - For IDENTIFYING which type of sign
   - Uses EfficientNet - state-of-the-art image classification
   - Balances accuracy and speed perfectly
   
3. TamperedSignDetector (ResNet/EfficientNet-based)
   - For VERIFYING if a sign is real or tampered
   - Binary classifier (Clean vs Tampered)
   - Uses proven CNN architectures

What is a Factory Pattern?
- Design pattern that creates objects without specifying exact class
- Like a car factory: you order "sedan" and get a complete car
- Makes code cleaner and more maintainable
"""

import torch  # PyTorch deep learning framework
import torch.nn as nn  # Neural network modules
from pathlib import Path  # For handling file paths
import os  # For file operations


class TrafficSignDetector:
    """
    YOLO TRAFFIC SIGN DETECTOR
    ==========================
    This class wraps a YOLO model for detecting traffic signs in images.
    
    YOLO (You Only Look Once):
    - Revolutionary detection algorithm from 2015
    - Looks at the image only once (hence the name)
    - Incredibly fast: can process video in real-time
    - Used in: self-driving cars, surveillance, robotics
    
    Why YOLO for traffic signs?
    - Speed: Need real-time detection for autonomous vehicles
    - Accuracy: Can detect multiple signs simultaneously
    - Robustness: Works in various lighting and weather conditions
    
    Model variations:
    - yolov8n: Nano (smallest, fastest, less accurate)
    - yolov8s: Small (balanced)
    - yolov8m: Medium (more accurate, slower)
    - yolov8l/x: Large/Extra large (most accurate, slowest)
    
    We use yolov8n by default for real-time performance.
    """
    
    def __init__(self, model_name='yolov8n'):
        """
        Initialize YOLO detector
        
        Args:
            model_name: Which YOLO variant to use (default: yolov8n - nano version)
        """
        self.model_name = model_name  # Store which YOLO version we're using
        self.model = None  # Will hold the actual YOLO model after loading
    
    def load_pretrained(self):
        """
        Load pre-trained YOLO model from Ultralytics
        
        Pre-trained means:
        - Model was already trained on millions of general images
        - We fine-tune it for traffic signs (transfer learning)
        - Much faster than training from scratch
        - Like hiring an experienced photographer and teaching them about traffic signs
        """
        try:
            from ultralytics import YOLO  # Import YOLO from Ultralytics library
            self.model = YOLO(f'{self.model_name}.pt')  # Load the model weights
            print(f"✅ Loaded pretrained {self.model_name} model")
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            raise
    
    def train(self, data_yaml, epochs=10, imgsz=640, batch=16, **kwargs):
        """Train the detection model"""
        if self.model is None:
            self.load_pretrained()
        
        # Filter kwargs to avoid unexpected arguments in different Ultralytics versions
        allowed_keys = {
            'device', 'workers', 'patience', 'cos_lr', 'optimizer', 'lr0', 'lrf',
            'momentum', 'weight_decay', 'label_smoothing', 'val', 'plots',
            'save_json', 'save_hybrid', 'verbose', 'seed', 'deterministic', 'amp',
            'cache', 'close_mosaic', 'rect', 'mosaic', 'mixup', 'copy_paste',
            'degrees', 'translate', 'scale', 'shear', 'perspective', 'flipud',
            'fliplr', 'hsv_h', 'hsv_s', 'hsv_v', 'accumulate', 'erasing',
            'auto_augment', 'copy_paste_mode', 'mask_ratio'
        }
        train_kwargs = {k: v for k, v in kwargs.items() if k in allowed_keys}

        # Ultralytics expects device as str or list sometimes; normalize int -> str
        if 'device' in train_kwargs and isinstance(train_kwargs['device'], int):
            train_kwargs['device'] = str(train_kwargs['device'])

        try:
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                save=True,
                project='runs/detect',
                name='traffic_sign_detection',
                **train_kwargs
            )
            return results
        except TypeError as e:
            # Retry with a minimal set if some keys still cause issues
            print(f"⚠️ YOLO.train received unsupported args, retrying with minimal set. Error: {e}")
            minimal_kwargs = {k: train_kwargs[k] for k in ['device', 'workers', 'amp', 'cache'] if k in train_kwargs}
            results = self.model.train(
                data=data_yaml,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                save=True,
                project='runs/detect',
                name='traffic_sign_detection',
                **minimal_kwargs
            )
            return results
    
    def save_model(self, path):
        """Save trained model"""
        if self.model:
            self.model.save(path)
            print(f"✅ Model saved to {path}")
    
    def load_model(self, path):
        """Load saved model"""
        if os.path.exists(path):
            from ultralytics import YOLO
            self.model = YOLO(path)
            print(f"✅ Model loaded from {path}")
            return True
        return False


class TamperedSignDetector(nn.Module):
    """Binary classifier for tampered/fake sign detection"""
    
    def __init__(self, model_name='efficientnet_b0'):
        super().__init__()
        self.model_name = model_name
        self.backbone = None
        self.classifier = None
        self._create_model()
    
    def _create_model(self):
        """Create the model architecture"""
        try:
            import timm
            self.backbone = timm.create_model(self.model_name, pretrained=True, num_classes=0)
            feature_dim = self.backbone.num_features
            self.classifier = nn.Linear(feature_dim, 2)  # Binary classification
            print(f"✅ Created {self.model_name} model")
        except Exception as e:
            print(f"❌ Error creating model: {e}")
            raise
    
    def forward(self, x):
        """Forward pass"""
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class TrafficSignClassifier:
    """Traffic sign classification model with advanced regularization"""
    
    def __init__(self, model_name='tf_efficientnetv2_s'):
        self.model_name = model_name
        self.model = None
    
    def create_model(self, num_classes=None, dropout_rate=0.3, drop_path_rate=0.2):
        """Create the classification model with regularization"""
        try:
            import timm
            # Use provided num_classes or default to 43 (your dataset size)
            classes = num_classes if num_classes is not None else 43
            
            # EfficientNetV2 models support these parameters
            self.model = timm.create_model(
                self.model_name, 
                pretrained=True, 
                num_classes=classes,
                drop_rate=dropout_rate,  # Dropout for regularization
                drop_path_rate=drop_path_rate  # Stochastic depth for better generalization
            )
            print(f"✅ Created {self.model_name} classifier with {classes} classes")
            print(f"   Regularization: dropout={dropout_rate}, drop_path={drop_path_rate}")
        except Exception as e:
            print(f"❌ Error creating classifier: {e}")
            raise


class ModelFactory:
    """Factory class for creating models"""
    
    @staticmethod
    def create_detection_model(model_name):
        """Create detection model"""
        return TrafficSignDetector(model_name)
    
    @staticmethod
    def create_fake_detector(model_name):
        """Create tampered detector model (kept for backward compatibility)"""
        return TamperedSignDetector(model_name)
    
    @staticmethod
    def create_tampered_detector(model_name):
        """Create tampered detector model"""
        return TamperedSignDetector(model_name)
    
    @staticmethod
    def create_classifier(model_name):
        """Create classifier model"""
        return TrafficSignClassifier(model_name)
