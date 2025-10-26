"""
Grad-CAM++ implementation (lightweight)
Generates class-discriminative localization maps for CNNs
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional


class GradCAMPPExplainer:
    """
    Minimal Grad-CAM++ for classification-like CNNs.
    """

    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.gradients = None
        self.activations = None
        self.hooks = []

    def explain_classification(self, model, input_tensor: torch.Tensor, target_layer: str, target_class: int) -> np.ndarray:
        """
        Generate Grad-CAM++ heatmap for classification model
        
        Args:
            model: PyTorch model
            input_tensor: Input tensor (preprocessed)
            target_layer: Target layer name for CAM
            target_class: Class index to explain
            
        Returns:
            Heatmap as numpy array (224x224)
        """
        model.eval()
        self._clear_hooks()
        
        # Auto-detect target layer if not specified or if set to 'auto'
        if target_layer is None or target_layer == 'auto':
            print(f"🔍 Auto-detecting last convolutional layer...")
            target_layer = self._find_last_conv_layer(model)
            if not target_layer:
                print("❌ No suitable convolutional layer found!")
                return np.ones((224, 224), dtype=np.float32) * 0.1
            print(f"✅ Using layer: {target_layer}")
        
        # Register hooks on the target layer
        if not self._register_hooks(model, target_layer):
            print(f"⚠️ Target layer '{target_layer}' not found, searching for last conv layer...")
            target_layer = self._find_last_conv_layer(model)
            if target_layer:
                print(f"✅ Using layer: {target_layer}")
                self._register_hooks(model, target_layer)
            else:
                print("❌ No suitable convolutional layer found!")
                return np.ones((224, 224), dtype=np.float32) * 0.1

        input_tensor = input_tensor.requires_grad_(True)
        scores = model(input_tensor)
        score = scores[:, target_class].sum()

        model.zero_grad()
        score.backward(retain_graph=True)

        heatmap = self._generate_cam_pp()
        self._clear_hooks()
        return heatmap
    
    def explain_detection(self, model, image_path: str, target_layer: str, roi_coords: tuple = None) -> Optional[np.ndarray]:
        """
        Generate Grad-CAM++ heatmap for detection model (simplified version)
        
        Args:
            model: YOLO detection model
            image_path: Path to input image
            target_layer: Target layer for CAM
            roi_coords: Region of interest coordinates (x1, y1, x2, y2)
            
        Returns:
            Heatmap as numpy array or None if failed
        """
        try:
            # For detection models, we return a simplified heatmap
            # Detection models are more complex and may require different handling
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                return None
            
            # Create a simple attention map (this is a placeholder)
            # In practice, detection models need specialized handling
            heatmap = np.ones((224, 224), dtype=np.float32) * 0.5
            
            if roi_coords is not None:
                x1, y1, x2, y2 = roi_coords
                # Highlight the ROI region in the heatmap
                h, w = image.shape[:2]
                hm_h, hm_w = 224, 224
                scale_x, scale_y = hm_w / w, hm_h / h
                roi_x1 = int(x1 * scale_x)
                roi_y1 = int(y1 * scale_y)
                roi_x2 = int(x2 * scale_x)
                roi_y2 = int(y2 * scale_y)
                heatmap[roi_y1:roi_y2, roi_x1:roi_x2] = 0.8
            
            return heatmap
        except Exception as e:
            print(f"Error in Grad-CAM++ detection: {e}")
            return None

    def _register_hooks(self, model, target_layer: str) -> bool:
        """Register hooks on target layer. Returns True if successful, False otherwise."""
        def forward_hook(module, inputs, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_module = None
        for name, module in model.named_modules():
            if name == target_layer or target_layer in name:
                target_module = module
                break

        if target_module is not None:
            self.hooks.append(target_module.register_forward_hook(forward_hook))
            self.hooks.append(target_module.register_full_backward_hook(backward_hook))
            return True
        return False
    
    def _find_last_conv_layer(self, model) -> Optional[str]:
        """
        Find the last convolutional layer in the model.
        Prioritizes layers in the backbone for models with that structure.
        """
        import torch.nn as nn
        
        # Check if model has a backbone attribute (common in timm models)
        if hasattr(model, 'backbone'):
            target_model = model.backbone
            prefix = 'backbone.'
        else:
            target_model = model
            prefix = ''
        
        last_conv_name = None
        last_conv_module = None
        
        # Find the last Conv2d layer with spatial dimensions
        for name, module in target_model.named_modules():
            if isinstance(module, nn.Conv2d):
                # Check if this conv layer has reasonable output channels
                if module.out_channels > 0:
                    last_conv_name = prefix + name
                    last_conv_module = module
        
        # Print info about the found layer
        if last_conv_name and last_conv_module:
            print(f"Found Conv2d layer: {last_conv_name} (out_channels={last_conv_module.out_channels})")
        
        return last_conv_name

    def _generate_cam_pp(self) -> np.ndarray:
        if self.gradients is None or self.activations is None:
            print(f"⚠️ Gradients or activations are None!")
            print(f"   Gradients: {self.gradients}")
            print(f"   Activations: {self.activations}")
            return np.ones((224, 224), dtype=np.float32) * 0.1

        gradients = self.gradients
        activations = self.activations
        
        print(f"📊 Grad-CAM++ Debug Info:")
        print(f"   Original gradients shape: {gradients.shape}")
        print(f"   Original activations shape: {activations.shape}")

        # Ensure tensors are at least 4D (batch, channel, height, width)
        while gradients.dim() < 4:
            gradients = gradients.unsqueeze(0)
            activations = activations.unsqueeze(0)
        
        print(f"   After unsqueeze - gradients: {gradients.shape}, activations: {activations.shape}")

        # Check if we have spatial dimensions (height and width)
        if gradients.dim() < 4 or gradients.size(-1) == 1 or gradients.size(-2) == 1:
            # If no spatial dimensions, create a simple uniform heatmap
            print(f"⚠️ No spatial dimensions! Using uniform heatmap")
            return np.ones((224, 224), dtype=np.float32) * 0.5

        grads2 = gradients.pow(2)
        grads3 = gradients.pow(3)

        eps = 1e-8
        # Use the last two dimensions for spatial aggregation
        spatial_dims = tuple(range(2, gradients.dim()))
        sum_acts = torch.sum(activations, dim=spatial_dims, keepdim=True)
        alphas_num = grads2
        alphas_denom = 2 * grads2 + sum_acts * grads3
        alphas = alphas_num / (alphas_denom + eps)

        relu_grads = F.relu(gradients)
        weights = torch.sum(alphas * relu_grads, dim=spatial_dims, keepdim=True)
        cam = torch.sum(weights * activations, dim=1)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        
        # Handle edge cases
        if cam.size == 0:
            print("⚠️ CAM size is 0!")
            return np.ones((224, 224), dtype=np.float32) * 0.5
        
        print(f"   CAM shape before normalization: {cam.shape}")
        print(f"   CAM value range: [{cam.min():.4f}, {cam.max():.4f}]")
        
        # Normalize
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
            print(f"   ✅ Normalized CAM range: [{cam.min():.4f}, {cam.max():.4f}]")
        else:
            print(f"⚠️ CAM has no variation (min={cam_min:.4f}, max={cam_max:.4f})")
            cam = np.ones_like(cam) * 0.5
        
        # Resize to 224x224 if not already
        if cam.ndim == 0:
            cam = np.ones((224, 224), dtype=np.float32) * 0.5
        elif cam.ndim == 1:
            # If 1D, create a simple gradient
            cam = np.tile(cam.reshape(-1, 1), (1, 224))
            cam = cv2.resize(cam, (224, 224))
        else:
            cam = cv2.resize(cam, (224, 224))
        
        print(f"   Final CAM shape: {cam.shape}, range: [{cam.min():.4f}, {cam.max():.4f}]")
        return cam

    def _clear_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.gradients = None
        self.activations = None


