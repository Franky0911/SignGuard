"""
Trainers package for SignGuard
Contains all model training modules

Note: Classification trainer removed - YOLO handles both detection and classification
"""

from .detection_trainer import DetectionTrainer
from .tampered_detector_trainer import TamperedDetectorTrainer

__all__ = [
    'DetectionTrainer',
    'TamperedDetectorTrainer'
]
