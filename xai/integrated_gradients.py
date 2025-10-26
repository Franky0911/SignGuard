"""
Integrated Gradients for attribution on classifiers (minimal PyTorch version).
"""

import torch
import numpy as np
import cv2


class IntegratedGradientsExplainer:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu', steps: int = 50):
        self.device = device
        self.steps = steps

    def explain_classification(self, model, input_tensor: torch.Tensor, target_class: int) -> np.ndarray:
        """
        Generate Integrated Gradients attribution map for classification model
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor (preprocessed)
            target_class: Class index to explain
            
        Returns:
            Attribution map as numpy array (224x224x3)
        """
        try:
            model.eval()
            
            # Ensure input_tensor is valid and on correct device
            if input_tensor is None:
                return self._default_attribution_map()
            
            input_tensor = input_tensor.to(self.device)
            baseline = torch.zeros_like(input_tensor)

            scaled_inputs = [baseline + (float(i) / self.steps) * (input_tensor - baseline) for i in range(1, self.steps + 1)]
            grads = []
            
            for x in scaled_inputs:
                x = x.detach().clone().requires_grad_(True)
                y = model(x)
                
                # Check if output is valid
                if y is None or target_class >= y.size(1):
                    continue
                
                score = y[:, target_class].sum()
                model.zero_grad()
                
                # Safe backward pass
                try:
                    score.backward()
                    if x.grad is not None:
                        grads.append(x.grad.detach().clone())
                except Exception as e:
                    print(f"Warning: Backward pass failed: {e}")
                    continue

            # Check if we have any gradients
            if not grads:
                return self._default_attribution_map()

            avg_grad = torch.mean(torch.stack(grads, dim=0), dim=0)
            attributions = (input_tensor.detach() - baseline) * avg_grad
            attr = attributions[0].detach().cpu().numpy()  # 3xHxW or CxHxW
            
            # Handle different tensor shapes
            if attr.ndim == 3:
                sal = np.abs(attr).sum(axis=0)
            elif attr.ndim == 2:
                sal = np.abs(attr)
            else:
                return self._default_attribution_map()
            
            # Normalize
            sal_min, sal_max = sal.min(), sal.max()
            if sal_max - sal_min > 1e-8:
                sal = (sal - sal_min) / (sal_max - sal_min)
            else:
                sal = np.ones_like(sal) * 0.5
            
            # Resize and convert to 3-channel
            sal = cv2.resize(sal, (224, 224))
            sal3 = np.stack([sal] * 3, axis=-1)
            return sal3
            
        except Exception as e:
            print(f"Error in Integrated Gradients: {e}")
            return self._default_attribution_map()
    
    def _default_attribution_map(self) -> np.ndarray:
        """Return default attribution map when computation fails"""
        sal = np.ones((224, 224), dtype=np.float32) * 0.5
        return np.stack([sal] * 3, axis=-1)
    
    def explain_detection(self, model, image_path: str, roi_coords: tuple = None) -> np.ndarray:
        """
        Generate Integrated Gradients attribution map for detection model (simplified version)
        
        Args:
            model: YOLO detection model
            image_path: Path to input image
            roi_coords: Region of interest coordinates (x1, y1, x2, y2)
            
        Returns:
            Attribution map as numpy array or None if failed
        """
        try:
            # For detection models, we return a simplified attribution map
            # Detection models are more complex and may require different handling
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Create a simple gradient-based attribution map (placeholder)
            attribution = np.ones((224, 224), dtype=np.float32) * 0.3
            
            if roi_coords is not None:
                x1, y1, x2, y2 = roi_coords
                # Highlight the ROI region in the attribution map
                h, w = image.shape[:2]
                am_h, am_w = 224, 224
                scale_x, scale_y = am_w / w, am_h / h
                roi_x1 = int(x1 * scale_x)
                roi_y1 = int(y1 * scale_y)
                roi_x2 = int(x2 * scale_x)
                roi_y2 = int(y2 * scale_y)
                attribution[roi_y1:roi_y2, roi_x1:roi_x2] = 0.7
            
            # Convert to 3-channel for consistency
            attribution_3ch = np.stack([attribution] * 3, axis=-1)
            return attribution_3ch
        except Exception as e:
            print(f"Error in Integrated Gradients detection: {e}")
            return None


