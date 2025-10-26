"""
Configuration loader and validator for SignGuard
"""

import os
import yaml
from pathlib import Path


class ConfigLoader:
    """Loads YAML config and provides validated paths with defaults."""

    def __init__(self, config_path: str = 'config/training_config.yaml'):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    def _load_config(self) -> dict:
        defaults = {
            'paths': {
                'processed_dir': 'dataset/processed',
                'augmented_dir': 'dataset/augmented',
                'fake_signs_dir': 'dataset/fake_signs'
            },
            'detection': {
                'model': 'yolov8n',
                'epochs': 5,
                'imgsz': 640,
                'batch_size': 16,
                # Advanced YOLO tuning (passed through to Ultralytics)
                'patience': 20,
                'cos_lr': True,
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'lrf': 0.01,
                'weight_decay': 0.0005,
                'label_smoothing': 0.0
            },
            'fake_detection': {
                'model': 'efficientnet_b0',
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'grad_clip_norm': 1.0,
                'early_stopping_patience': 5,
                'scheduler': 'cosine',
                'scheduler_params': {
                    'T_max': 10,
                    'eta_min': 1e-5
                }
            },
            'classification': {
                'model': 'efficientnet_b3',
                'epochs': 5,
                'batch_size': 16,
                'learning_rate': 1e-3,
                'weight_decay': 1e-4,
                'grad_clip_norm': 1.0,
                'early_stopping_patience': 5,
                'scheduler': 'cosine',
                'scheduler_params': {
                    'T_max': 10,
                    'eta_min': 1e-5
                }
            },
            'synthetic': {
                'num_fake_images': 1000
            }
        }

        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                user_cfg = yaml.safe_load(f) or {}
        else:
            user_cfg = {}

        # Merge user config over defaults (shallow for simplicity)
        def merge(a, b):
            out = dict(a)
            for k, v in (b or {}).items():
                if isinstance(v, dict) and isinstance(out.get(k), dict):
                    out[k] = merge(out[k], v)
                else:
                    out[k] = v
            return out

        return merge(defaults, user_cfg)

    def validate_paths(self) -> None:
        paths = self.config.get('paths', {})
        # Ensure required directories exist
        for key in ['processed_dir', 'augmented_dir', 'fake_signs_dir']:
            path = Path(paths.get(key, ''))
            if not path:
                continue
            if key in ['fake_signs_dir']:
                path.mkdir(parents=True, exist_ok=True)
            else:
                # Do not create dataset folders implicitly, but warn
                if not path.exists():
                    print(f"⚠️  Path does not exist: {path}")

"""
Configuration loader utility for SignGuard
Handles loading and validation of configuration files
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """Handles configuration loading and validation"""
    
    def __init__(self, config_path: str = "config/training_config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file with defaults"""
        default_config = {
            'detection': {
                'model': 'yolov8n',
                'epochs': 10,
                'imgsz': 640,
                'batch_size': 16,
                'learning_rate': 0.01,
                'patience': 5,
                'save_period': 5
            },
            'fake_detection': {
                'model': 'efficientnet_b0',
                'epochs': 5,
                'batch_size': 32,
                'learning_rate': 0.001,
                'patience': 3,
                'num_classes': 2
            },
            'classification': {
                'model': 'efficientnet_b3',
                'epochs': 5,
                'batch_size': 32,
                'learning_rate': 0.001,
                'patience': 3,
                'num_classes': 74
            },
            'data': {
                'train_split': 0.8,
                'val_split': 0.2,
                'num_workers': 2,
                'pin_memory': True
            },
            'synthetic': {
                'num_fake_images': 2000,
                'modifications_per_image': 3,
                'noise_level': 25,
                'color_shift_range': 20
            },
            'paths': {
                'models_dir': 'models',
                'data_dir': 'dataset',
                'augmented_dir': 'dataset/augmented',
                'fake_signs_dir': 'dataset/fake_signs',
                'logs_dir': 'logs'
            }
        }
        
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    user_config = yaml.safe_load(f)
                    # Deep merge user config with defaults
                    self._deep_merge(default_config, user_config)
            except Exception as e:
                print(f"⚠️  Warning: Could not load config from {self.config_path}: {e}")
                print("Using default configuration")
        
        return default_config
    
    def _deep_merge(self, base_dict: Dict, update_dict: Dict) -> None:
        """Deep merge update_dict into base_dict"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._deep_merge(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key_path: str, default=None):
        """Get configuration value using dot notation (e.g., 'detection.epochs')"""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config.get(section, {})
    
    def validate_paths(self) -> bool:
        """Validate that required paths exist"""
        paths = self.get_section('paths')
        required_dirs = ['models_dir', 'data_dir', 'fake_signs_dir']
        
        for dir_key in required_dirs:
            dir_path = paths.get(dir_key)
            if dir_path and not os.path.exists(dir_path):
                print(f"⚠️  Creating directory: {dir_path}")
                Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        return True
    
    def save_config(self, output_path: str = None) -> None:
        """Save current configuration to file"""
        if output_path is None:
            output_path = self.config_path
        
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        print(f"✅ Configuration saved to {output_path}")
