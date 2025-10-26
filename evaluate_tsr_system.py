"""
Comprehensive TSR System Evaluation with XAI Integration
Tests detection, classification, and fake detection models with explainable AI
"""

import os
import cv2
import torch
import numpy as np
from pathlib import Path
import yaml
from ultralytics import YOLO
from PIL import Image
import torchvision.transforms as transforms

from models import ModelFactory
from utils.data_utils import get_fake_detection_loaders, get_classification_loaders

# Import XAI components
try:
    from xai import XAIExplainer
    XAI_AVAILABLE = True
except ImportError:
    XAI_AVAILABLE = False
    print("⚠️  XAI not available. Install XAI dependencies for full functionality.")


class TSRSystemEvaluator:
    """Comprehensive TSR system evaluator with XAI integration"""
    
    def __init__(self, config_path='config/training_config.yaml', enable_xai=True):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load models
        self.detection_model = None  # YOLO handles both detection AND classification
        self.fake_detection_model = None  # Renamed to tampered_detection_model
        
        # Initialize XAI if available
        self.xai_explainer = None
        self.enable_xai = enable_xai and XAI_AVAILABLE
        
        if self.enable_xai:
            try:
                self.xai_explainer = XAIExplainer(self.device)
                print("✅ XAI explainer initialized")
            except Exception as e:
                print(f"⚠️  XAI initialization failed: {e}")
                self.enable_xai = False
        
        self._load_models()
    
    def _load_models(self):
        """Load all trained models"""
        print("Loading models...")
        
        # Load detection model
        detection_paths = [
            'runs/detect/traffic_sign_detection/weights/best.pt',
            'runs/detect/traffic_sign_detection/best.pt',
            'models/traffic_sign_detection.pt'
        ]
        
        for detection_path in detection_paths:
            if os.path.exists(detection_path):
                self.detection_model = YOLO(detection_path)
                print(f"✅ Detection model loaded from {detection_path}")
                break
        else:
            print("⚠️  Detection model not found in any expected location")
        
        # Classification model removed - YOLO already handles classification
        print("ℹ️  Classification is handled by YOLO detection model")
        
        # Load fake detection model
        fake_detection_path = 'models/fake_sign_detector.pth'
        if os.path.exists(fake_detection_path):
            try:
                self.fake_detection_model = ModelFactory.create_fake_detector(
                    self.config['fake_detection']['model']
                )
                # Load the model state dict directly
                state_dict = torch.load(fake_detection_path, map_location=self.device)
                self.fake_detection_model.load_state_dict(state_dict)
                self.fake_detection_model.to(self.device)
                self.fake_detection_model.eval()
                print("✅ Fake detection model loaded")
            except Exception as e:
                print(f"⚠️  Fake detection model loading failed: {e}")
                self.fake_detection_model = None
    
    def evaluate_image(self, image_path, enable_xai_explanations=False):
        """Evaluate a single image through the complete TSR pipeline"""
        print(f"\n🔍 Evaluating: {image_path}")
        
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            print("❌ Could not load image")
            return None
        
        results = {
            'image_path': str(image_path),
            'detections': [],  # YOLO detections with class_id (classification built-in)
            'fake_predictions': [],  # Tampered/fake detection
            'xai_explanations': {} if enable_xai_explanations else None
        }
        
        # 1. Detection - Use improved detection if available
        if self.detection_model:
            try:
                # Try improved detection first
                from utils.improved_detection import create_improved_detector
                improved_detector = create_improved_detector()
                detections = improved_detector.detect(image)
                
                for detection in detections:
                    results['detections'].append({
                        'bbox': detection['bbox'],
                        'confidence': detection['confidence'],
                        'class_id': detection['class_id']
                    })
                
                print(f"✅ Improved detection found {len(detections)} objects")
                
            except Exception as e:
                # Fallback to standard detection
                print(f"⚠️ Using standard detection (improved detection failed: {e})")
                detection_results = self.detection_model(image)
                for result in detection_results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            
                            results['detections'].append({
                                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                'confidence': float(conf),
                                'class_id': cls
                            })
        
        # 2. Fake Detection for each detection (Classification already done by YOLO)
        if results['detections'] and self.fake_detection_model:
            for i, detection in enumerate(results['detections']):
                x1, y1, x2, y2 = detection['bbox']
                
                # Extract sign region
                sign_region = image[y1:y2, x1:x2]
                if sign_region.size == 0:
                    continue
                
                # Fake/Tampered Detection
                if self.fake_detection_model:
                    fake_pred = self._detect_fake(sign_region)
                    fake_result = {
                        'detection_id': i,
                        'is_fake': fake_pred['is_fake'],
                        'confidence': fake_pred['confidence']
                    }
                    
                    # Add XAI explanations if enabled
                    if enable_xai_explanations and self.enable_xai and self.xai_explainer:
                        try:
                            print(f"  🔍 Generating XAI explanations for fake detection {i+1}...")
                            xai_explanation = self.xai_explainer.explain_fake_detection(
                                self.fake_detection_model.model, sign_region, 
                                methods=['gradcam', 'shap', 'lime']
                            )
                            fake_result['xai_explanation'] = xai_explanation
                        except Exception as e:
                            print(f"    ⚠️  XAI fake detection failed: {e}")
                    
                    results['fake_predictions'].append(fake_result)
        
        return results
    
    def _get_class_names(self):
        """Get class names from configuration"""
        try:
            with open('dataset/augmented/augmented_config.yaml', 'r') as f:
                config = yaml.safe_load(f)
            return config['names']
        except:
            return [f"Class_{i}" for i in range(43)]  # Default fallback
    
    # Classification method removed - YOLO detection already includes classification
    
    def _detect_fake(self, sign_region):
        """Detect if a sign is fake"""
        if self.fake_detection_model is None:
            return {'is_fake': False, 'confidence': 0.5}
            
        # Resize and preprocess
        sign_pil = Image.fromarray(cv2.cvtColor(sign_region, cv2.COLOR_BGR2RGB))
        
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform(sign_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.fake_detection_model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            fake_prob = probabilities[0][1].item()  # Probability of being fake
            is_fake = fake_prob > 0.5
            
            return {
                'is_fake': is_fake,
                'confidence': fake_prob if is_fake else 1 - fake_prob
            }
    
    def evaluate_dataset(self, dataset_path, output_file='evaluation_results.json'):
        """Evaluate a dataset of images"""
        import json
        
        dataset_path = Path(dataset_path)
        image_files = list(dataset_path.glob('*.jpg')) + list(dataset_path.glob('*.png'))
        
        print(f"Evaluating {len(image_files)} images...")
        
        all_results = []
        for i, img_path in enumerate(image_files):
            if i % 10 == 0:
                print(f"Progress: {i}/{len(image_files)}")
            
            result = self.evaluate_image(img_path)
            if result:
                all_results.append(result)
        
        # Save results
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"✅ Evaluation complete. Results saved to {output_file}")
        return all_results
    
    def print_summary(self, results):
        """Print evaluation summary"""
        print("\n📊 EVALUATION SUMMARY")
        print("=" * 50)
        
        total_images = len(results)
        total_detections = sum(len(r['detections']) for r in results)
        total_fake_predictions = sum(len(r['fake_predictions']) for r in results)
        
        fake_detections = sum(1 for r in results for fp in r['fake_predictions'] if fp['is_fake'])
        
        print(f"Total images processed: {total_images}")
        print(f"Total detections (with YOLO classification): {total_detections}")
        print(f"Total fake predictions: {total_fake_predictions}")
        print(f"Fake signs detected: {fake_detections}")
        
        if total_detections > 0:
            print(f"Average detections per image: {total_detections/total_images:.2f}")
        if total_fake_predictions > 0:
            print(f"Fake detection rate: {fake_detections/total_fake_predictions:.2%}")


def main():
    """Main evaluation function with XAI support"""
    # Initialize evaluator with XAI enabled
    evaluator = TSRSystemEvaluator(enable_xai=True)
    
    # Evaluate test dataset
    test_path = 'dataset/augmented/test/images'
    if os.path.exists(test_path):
        print("Evaluating test dataset...")
        results = evaluator.evaluate_dataset(test_path)
        evaluator.print_summary(results)
    else:
        print("Test dataset not found. Please run training first.")
    
    # Evaluate a single image if provided
    import sys
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            # Evaluate with XAI explanations
            print(f"\n🔍 Evaluating single image with XAI: {image_path}")
            result = evaluator.evaluate_image(image_path, enable_xai_explanations=True)
            if result:
                print("\nSingle image evaluation:")
                print(f"Detections (with YOLO classification): {len(result['detections'])}")
                print(f"Fake predictions: {len(result['fake_predictions'])}")
                
                # Show detection details
                for i, det in enumerate(result['detections']):
                    print(f"  Detection {i+1}:")
                    print(f"    Class ID: {det['class_id']}")
                    print(f"    Confidence: {det['confidence']:.3f}")
                
                # Show fake detection XAI results (classification XAI removed)
                for i, fake_pred in enumerate(result['fake_predictions']):
                    if 'xai_explanation' in fake_pred:
                        xai_exp = fake_pred['xai_explanation']
                        print(f"  Fake Detection {i+1} XAI:")
                        print(f"    Prediction: {'FAKE' if xai_exp['prediction']['is_fake'] else 'REAL'}")
                        print(f"    Confidence: {xai_exp['prediction']['confidence']:.3f}")
        else:
            print(f"Image not found: {image_path}")
    else:
        print("\n💡 Usage: python evaluate_tsr_system.py <image_path>")
        print("   Example: python evaluate_tsr_system.py dataset/augmented/test/images/00000.png")


if __name__ == "__main__":
    main()
