"""
Detection Model Trainer
Handles training of the main traffic sign detection model
"""

import os
import time
from pathlib import Path
from models import ModelFactory


class DetectionTrainer:
    """Handles detection model training"""
    
    def __init__(self, config):
        self.config = config
        self.detection_model = None
    
    def train(self, data_yaml):
        """Train the detection model"""
        model_path = 'models/traffic_sign_detection.pt'
        
        # Check if model already exists
        if os.path.exists(model_path):
            print("✅ Detection model already exists, skipping training")
            print(f"📁 Model found at: {model_path}")
            return {'status': 'skipped', 'model_path': model_path}
        
        print("🚀 Training Traffic Sign Detection Model")
        print("=" * 50)
        
        try:
            # Create detection model using full model name from config (e.g., 'yolov8n')
            self.detection_model = ModelFactory.create_detection_model(
                self.config['detection']['model']
            )
            
            # Load pretrained model
            self.detection_model.load_pretrained()
            
            # Train model
            det_cfg = self.config['detection']
            results = self.detection_model.train(
                data_yaml=data_yaml,
                epochs=det_cfg['epochs'],
                imgsz=det_cfg['imgsz'],
                batch=det_cfg['batch_size'],
                patience=det_cfg.get('patience', 20),
                cos_lr=det_cfg.get('cos_lr', True),
                optimizer=det_cfg.get('optimizer', 'AdamW'),
                lr0=det_cfg.get('lr0', 0.001),
                lrf=det_cfg.get('lrf', 0.01),
                weight_decay=det_cfg.get('weight_decay', 0.0005),
                label_smoothing=det_cfg.get('label_smoothing', 0.0),
                device=det_cfg.get('device', 0),
                workers=det_cfg.get('workers', 4),
                amp=det_cfg.get('amp', True),
                cache=det_cfg.get('cache', False),
                close_mosaic=det_cfg.get('close_mosaic', 0),
                rect=det_cfg.get('rect', False),
                overlap_mask=det_cfg.get('overlap_mask', False),
                mask_ratio=det_cfg.get('mask_ratio', 4),
                dropout=det_cfg.get('dropout', 0.0),
                val=det_cfg.get('val', True),
                plots=det_cfg.get('plots', False),
                save_json=det_cfg.get('save_json', False),
                save_hybrid=det_cfg.get('save_hybrid', False),
                verbose=det_cfg.get('verbose', False),
                seed=det_cfg.get('seed', 0),
                deterministic=det_cfg.get('deterministic', False),
                bench=det_cfg.get('benchmark', True),
                compile=det_cfg.get('compile', False),
                mosaic=det_cfg.get('mosaic', 1.0),
                mixup=det_cfg.get('mixup', 0.0),
                copy_paste=det_cfg.get('copy_paste', 0.0),
                copy_paste_mode=det_cfg.get('copy_paste_mode', 'flip'),
                auto_augment=det_cfg.get('auto_augment', None),
                erasing=det_cfg.get('erasing', 0.2),
                degrees=det_cfg.get('degrees', 0.0),
                translate=det_cfg.get('translate', 0.0),
                scale=det_cfg.get('scale', 0.5),
                shear=det_cfg.get('shear', 0.0),
                perspective=det_cfg.get('perspective', 0.0),
                flipud=det_cfg.get('flipud', 0.0),
                fliplr=det_cfg.get('fliplr', 0.5),
                hsv_h=det_cfg.get('hsv_h', 0.0),
                hsv_s=det_cfg.get('hsv_s', 0.0),
                hsv_v=det_cfg.get('hsv_v', 0.0)
            )
            
            # Save model
            Path('models').mkdir(exist_ok=True)
            self.detection_model.save_model(model_path)
            
            print(f"✅ Detection model trained and saved to {model_path}")
            return {'status': 'completed', 'model_path': model_path, 'results': results}
            
        except Exception as e:
            print(f"❌ Detection training failed: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def load_model(self, model_path):
        """Load a trained detection model"""
        if os.path.exists(model_path):
            self.detection_model = ModelFactory.create_detection_model(
                self.config['detection']['model']
            )
            self.detection_model.load_model(model_path)
            return True
        return False
