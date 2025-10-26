from pathlib import Path
from typing import Dict, Optional, List

from xai import XAIExplainer


def get_xai_explanations(model, image_path: str, model_type: str, class_names: Optional[List[str]] = None) -> Dict:
    """
    Get XAI explanations for different model types
    
    Note: Classification XAI removed - YOLO handles both detection and classification
    """
    explainer = XAIExplainer()
    if model_type == 'detection':
        return explainer.explain_detection(model, image_path)
    if model_type == 'fake_detection' or model_type == 'tampered_detection':
        # Use fake_detection explainer for both (they are binary classifiers)
        return explainer.explain_fake_detection(model, image_path)
    # Classification model type removed - YOLO detection includes classification
    return {}


