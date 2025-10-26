"""
Improved Detection Utilities for Better Performance on Unseen Data
Handles various edge cases and provides robust detection strategies
"""

import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import yaml
from ultralytics import YOLO


class ImprovedDetector:
    """Enhanced detection class with multiple fallback strategies"""
    
    def __init__(self, model_path: str, config_path: str = 'config/detection_config.yaml'):
        self.model = YOLO(model_path)
        self.config = self._load_config(config_path)
        self.class_names = self._load_class_names()
        
    def _load_config(self, config_path: str) -> Dict:
        """Load detection configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return {
                'detection': {
                    'confidence_threshold': 0.15,
                    'min_box_size': 5,
                    'fallback_threshold': 0.05,
                    'progressive_thresholds': [0.1, 0.05, 0.01, 0.005],
                    'multi_scale_detection': True,
                    'input_size': 640,
                    'iou_threshold': 0.45,
                    'max_detections': 100
                }
            }
    
    def _load_class_names(self) -> List[str]:
        """Load class names with fallback strategies"""
        # Try multiple sources for class names
        class_sources = [
            'config/class_names.txt',
            'config/class_names.json',
            'dataset/augmented/augmented_config.yaml'
        ]
        
        for source in class_sources:
            try:
                if source.endswith('.txt'):
                    with open(source, 'r', encoding='utf-8') as f:
                        names = [line.strip() for line in f.readlines() if line.strip()]
                        if names:
                            return names
                elif source.endswith('.json'):
                    import json
                    with open(source, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if isinstance(data, list) and data:
                            return data
                elif source.endswith('.yaml'):
                    with open(source, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        names = config.get('names', [])
                        if names:
                            return names
            except Exception:
                continue
        
        # Fallback to generic names
        return [f"Class_{i}" for i in range(43)]
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Enhanced image preprocessing for better detection"""
        # Convert to RGB if needed
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if it's BGR and convert to RGB
            if image.dtype == np.uint8:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure proper data type
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        
        # Apply histogram equalization for better contrast
        if len(image.shape) == 3:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            image[:, :, 0] = clahe.apply(image[:, :, 0])
            image = cv2.cvtColor(image, cv2.COLOR_LAB2RGB)
        
        return image
    
    def detect_with_progressive_thresholds(self, image: np.ndarray) -> List[Dict]:
        """Detect objects using progressive confidence thresholds"""
        detections = []
        thresholds = self.config['detection']['progressive_thresholds']
        
        for threshold in thresholds:
            try:
                results = self.model(image, conf=threshold, iou=self.config['detection']['iou_threshold'])
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        clss = result.boxes.cls.cpu().numpy()
                        
                        for box, conf, cls in zip(boxes, confs, clss):
                            x1, y1, x2, y2 = box.astype(int)
                            width = x2 - x1
                            height = y2 - y1
                            
                            # Check minimum size
                            if width >= self.config['detection']['min_box_size'] and height >= self.config['detection']['min_box_size']:
                                detection = {
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(conf),
                                    'class_id': int(cls),
                                    'class_name': self.class_names[int(cls)] if int(cls) < len(self.class_names) else f"Class_{int(cls)}",
                                    'threshold_used': threshold
                                }
                                detections.append(detection)
                
                # If we found detections, break
                if detections:
                    break
                    
            except Exception as e:
                print(f"Detection failed with threshold {threshold}: {e}")
                continue
        
        return detections
    
    def detect_multi_scale(self, image: np.ndarray) -> List[Dict]:
        """Detect objects at multiple scales for better coverage"""
        detections = []
        scales = [0.8, 1.0, 1.2]  # Different scales
        
        for scale in scales:
            try:
                # Resize image
                h, w = image.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                scaled_image = cv2.resize(image, (new_w, new_h))
                
                # Detect on scaled image
                results = self.model(scaled_image, conf=self.config['detection']['confidence_threshold'])
                
                for result in results:
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confs = result.boxes.conf.cpu().numpy()
                        clss = result.boxes.cls.cpu().numpy()
                        
                        for box, conf, cls in zip(boxes, confs, clss):
                            # Scale back to original coordinates
                            x1, y1, x2, y2 = (box / scale).astype(int)
                            width = x2 - x1
                            height = y2 - y1
                            
                            if width >= self.config['detection']['min_box_size'] and height >= self.config['detection']['min_box_size']:
                                detection = {
                                    'bbox': [x1, y1, x2, y2],
                                    'confidence': float(conf),
                                    'class_id': int(cls),
                                    'class_name': self.class_names[int(cls)] if int(cls) < len(self.class_names) else f"Class_{int(cls)}",
                                    'scale_used': scale
                                }
                                detections.append(detection)
                                
            except Exception as e:
                print(f"Multi-scale detection failed at scale {scale}: {e}")
                continue
        
        return detections
    
    def detect_robust(self, image: np.ndarray) -> List[Dict]:
        """Robust detection with multiple strategies"""
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        all_detections = []
        
        # Strategy 1: Progressive thresholds
        if self.config['detection'].get('progressive_thresholds'):
            detections = self.detect_with_progressive_thresholds(processed_image)
            all_detections.extend(detections)
        
        # Strategy 2: Multi-scale detection
        if self.config['detection'].get('multi_scale_detection', False):
            detections = self.detect_multi_scale(processed_image)
            all_detections.extend(detections)
        
        # Strategy 3: Standard detection with lower threshold
        try:
            results = self.model(processed_image, conf=self.config['detection']['fallback_threshold'])
            for result in results:
                if result.boxes is not None:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confs = result.boxes.conf.cpu().numpy()
                    clss = result.boxes.cls.cpu().numpy()
                    
                    for box, conf, cls in zip(boxes, confs, clss):
                        x1, y1, x2, y2 = box.astype(int)
                        width = x2 - x1
                        height = y2 - y1
                        
                        if width >= self.config['detection']['min_box_size'] and height >= self.config['detection']['min_box_size']:
                            detection = {
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(conf),
                                'class_id': int(cls),
                                'class_name': self.class_names[int(cls)] if int(cls) < len(self.class_names) else f"Class_{int(cls)}",
                                'strategy': 'standard'
                            }
                            all_detections.append(detection)
        except Exception as e:
            print(f"Standard detection failed: {e}")
        
        # Remove duplicates and return best detections
        return self._remove_duplicates(all_detections)
    
    def _remove_duplicates(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Remove duplicate detections based on IoU"""
        if not detections:
            return []
        
        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        for detection in detections:
            is_duplicate = False
            for existing in filtered:
                if self._calculate_iou(detection['bbox'], existing['bbox']) > iou_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                filtered.append(detection)
        
        return filtered
    
    def _calculate_iou(self, box1: List[int], box2: List[int]) -> float:
        """Calculate Intersection over Union (IoU) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """Main detection method with all strategies"""
        try:
            return self.detect_robust(image)
        except Exception as e:
            print(f"Detection failed: {e}")
            return []


def create_improved_detector(model_path: str = 'runs/detect/traffic_sign_detection/best.pt') -> ImprovedDetector:
    """Factory function to create an improved detector"""
    return ImprovedDetector(model_path)
