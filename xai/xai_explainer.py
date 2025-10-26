"""
EXPLAINABLE AI (XAI) MODULE - MAKING AI DECISIONS TRANSPARENT
==============================================================
This module answers the question: "WHY did the AI make this decision?"

What is Explainable AI (XAI)?
- Normal AI: "This is a stop sign" (black box - we don't know why)
- XAI: "This is a stop sign BECAUSE of these red regions and this octagonal shape"
  (transparent - we can see the reasoning)

Why is this important?
- Trust: Users need to trust AI decisions, especially in safety-critical applications
- Debugging: If AI makes mistakes, we can see what it's looking at
- Safety: For self-driving cars, we MUST understand why AI makes decisions
- Legal: In some domains, AI decisions must be explainable by law

Our XAI Methods:
1. Grad-CAM++ (Gradient-weighted Class Activation Mapping)
   - Shows WHICH parts of the image were most important for the decision
   - Creates a heatmap highlighting important regions
   - Like highlighting text with a marker to show what's important

2. Integrated Gradients
   - Shows HOW MUCH each pixel contributed to the decision
   - More precise than Grad-CAM++, but slower
   - Like showing exactly which ingredients make a dish taste good

Think of XAI as "showing your work" in math class - not just the answer, but how you got there!
"""

import torch  # PyTorch for deep learning
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For creating visualizations
import cv2  # OpenCV for image processing
from pathlib import Path  # For file path handling
from typing import Dict, List, Tuple, Optional, Union  # For type hints
import warnings  # For suppressing warning messages
warnings.filterwarnings('ignore')  # Suppress warnings for cleaner output

# Import our specific XAI implementations
from .gradcam_pp import GradCAMPPExplainer  # Grad-CAM++ for visual explanations
from .integrated_gradients import IntegratedGradientsExplainer  # Integrated Gradients for attribution


class XAIExplainer:
    """
    UNIFIED XAI INTERFACE
    =====================
    This class is the main controller for all explainability methods.
    It works with all three of our AI models:
    1. Detection Model (YOLO)
    2. Classification Model
    3. Tampering Detector
    
    What it does:
    - Takes an AI model and an image
    - Runs the prediction
    - Generates visual explanations showing WHY the AI decided what it did
    - Creates heatmaps and charts for easy understanding
    - Saves results for later review
    
    Think of this as a "lie detector" for AI - it reveals what the AI is thinking!
    """
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the XAI Explainer
        
        Args:
            device: Computing device ('cuda' for GPU, 'cpu' for CPU)
                   GPU is much faster for generating explanations
        """
        self.device = device  # Store which computing device to use
        # Create the two explanation methods
        self.gradcam_pp = GradCAMPPExplainer(device)  # For heatmap visualizations
        self.integrated_gradients = IntegratedGradientsExplainer(device)  # For pixel attribution
        
        # Create folders to save explanation results
        self.output_dir = Path('xai_outputs')
        self.output_dir.mkdir(exist_ok=True)  # Main output folder
        (self.output_dir / 'detection').mkdir(exist_ok=True)  # Detection explanations
        (self.output_dir / 'tampered_detection').mkdir(exist_ok=True)  # Tampered detection explanations
    
    def explain_detection(self, model, image_path: str, target_layer: str = 'model.22.cv3') -> Dict:
        """
        Explain YOLO detection model predictions using Grad-CAM++ and Integrated Gradients
        
        Args:
            model: Trained YOLO model
            image_path: Path to input image
            target_layer: Target layer for Grad-CAM++
            
        Returns:
            Dictionary with explanations and visualizations
        """
        print("🔍 Explaining detection model with Grad-CAM++ and Integrated Gradients...")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Could not load image from {image_path}")
            return {
                'image_path': image_path,
                'predictions': [],
                'gradcam_pp_maps': [],
                'integrated_gradients_maps': []
            }
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get YOLO predictions
        try:
            results = model(image_path)  # YOLO models are callable directly
        except Exception as e:
            print(f"Error running YOLO detection: {e}")
            return {
                'image_path': image_path,
                'predictions': [],
                'gradcam_pp_maps': [],
                'integrated_gradients_maps': []
            }
        
        explanations = {
            'image_path': image_path,
            'predictions': [],
            'gradcam_pp_maps': [],
            'integrated_gradients_maps': []
        }
        
        # Process each detection
        for i, result in enumerate(results):
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy()
                
                for j, (box, conf, cls) in enumerate(zip(boxes, confidences, classes)):
                    x1, y1, x2, y2 = box.astype(int)
                    
                    # Extract ROI for explanation
                    roi = image_rgb[y1:y2, x1:x2]
                    
                    if roi.size > 0:
                        # Grad-CAM++ for detection
                        try:
                            gradcam_pp_map = self.gradcam_pp.explain_detection(
                                model, image_path, target_layer, 
                                roi_coords=(x1, y1, x2, y2)
                            )
                        except Exception as e:
                            print(f"Warning: Grad-CAM++ failed: {e}")
                            gradcam_pp_map = None
                        
                        # Integrated Gradients for detection
                        try:
                            ig_map = self.integrated_gradients.explain_detection(
                                model, image_path, roi_coords=(x1, y1, x2, y2)
                            )
                        except Exception as e:
                            print(f"Warning: Integrated Gradients failed: {e}")
                            ig_map = None
                        
                        detection_info = {
                            'bbox': box.tolist(),
                            'confidence': float(conf),
                            'class_id': int(cls),
                            'gradcam_pp_map': gradcam_pp_map,
                            'integrated_gradients_map': ig_map
                        }
                        
                        explanations['predictions'].append(detection_info)
                        explanations['gradcam_pp_maps'].append(gradcam_pp_map)
                        explanations['integrated_gradients_maps'].append(ig_map)
        
        # Create visualization
        try:
            self._visualize_detection_explanations(image_rgb, explanations)
            print(f"✅ Detection XAI visualization saved to: {self.output_dir / 'detection'}")
        except Exception as e:
            print(f"Warning: Failed to create detection visualization: {e}")
        
        return explanations
    
    def explain_classification(self, model, image_path: str, class_names: List[str], 
                             target_layer: str = 'classifier') -> Dict:
        """
        Explain classification model predictions using Grad-CAM++ and Integrated Gradients
        
        Args:
            model: Trained classification model
            image_path: Path to input image
            class_names: List of class names
            target_layer: Target layer for Grad-CAM++
            
        Returns:
            Dictionary with explanations
        """
        print("🔍 Explaining classification model with Grad-CAM++ and Integrated Gradients...")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            # Preprocess image
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image_rgb).unsqueeze(0)
            # Ensure input tensor is on the same device as model
            model_device = next(model.parameters()).device
            input_tensor = input_tensor.to(model_device)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Generate explanations with Grad-CAM++
        try:
            gradcam_pp_map = self.gradcam_pp.explain_classification(
                model, input_tensor, target_layer, predicted_class
            )
        except Exception as e:
            print(f"Warning: Grad-CAM++ failed: {e}")
            gradcam_pp_map = None

        # Integrated Gradients
        try:
            ig_map = self.integrated_gradients.explain_classification(
                model, input_tensor, predicted_class
            )
        except Exception as e:
            print(f"Warning: Integrated Gradients failed: {e}")
            ig_map = None

        explanations = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_class_name': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': probabilities[0].cpu().numpy().tolist(),
            'gradcam_pp_map': gradcam_pp_map,
            'integrated_gradients': ig_map
        }
        
        # Create visualization
        self._visualize_classification_explanations(image_rgb, explanations, class_names)
        
        return explanations
    
    def explain_fake_detection(self, model, image_path: str, 
                             target_layer: str = None) -> Dict:
        """
        Explain tampered detection model predictions using Grad-CAM++ and Integrated Gradients
        
        Args:
            model: Trained tampered detection model
            image_path: Path to input image
            target_layer: Target layer for Grad-CAM++
            
        Returns:
            Dictionary with explanations
        """
        print("🔍 Explaining tampered detection model with Grad-CAM++ and Integrated Gradients...")
        
        # Load and preprocess image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Get prediction
        model.eval()
        with torch.no_grad():
            from torchvision import transforms
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            input_tensor = transform(image_rgb).unsqueeze(0)
            # Ensure input tensor is on the same device as model
            model_device = next(model.parameters()).device
            input_tensor = input_tensor.to(model_device)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # Grad-CAM++
        try:
            # If no target layer specified, auto-detection will be used
            gradcam_pp_map = self.gradcam_pp.explain_classification(
                model, input_tensor, target_layer, predicted_class
            )
        except Exception as e:
            print(f"Warning: Grad-CAM++ failed: {e}")
            import traceback
            traceback.print_exc()
            gradcam_pp_map = None

        # Integrated Gradients
        try:
            ig_map = self.integrated_gradients.explain_classification(
                model, input_tensor, predicted_class
            )
        except Exception as e:
            print(f"Warning: Integrated Gradients failed: {e}")
            ig_map = None
        
        class_names = ['Clean', 'Tampered']

        explanations = {
            'image_path': image_path,
            'predicted_class': predicted_class,
            'predicted_class_name': class_names[predicted_class],
            'confidence': confidence,
            'clean_probability': probabilities[0][0].item(),
            'tampered_probability': probabilities[0][1].item(),
            'gradcam_pp_map': gradcam_pp_map,
            'integrated_gradients': ig_map
        }
        
        # Create visualization
        try:
            self._visualize_tampered_detection_explanations(image_rgb, explanations)
            print(f"✅ Tampered detection XAI visualization saved to: {self.output_dir / 'tampered_detection'}")
        except Exception as e:
            print(f"Warning: Failed to create tampered detection visualization: {e}")
        
        return explanations
    
    def _visualize_detection_explanations(self, image: np.ndarray, explanations: Dict):
        """Create visualization for detection explanations with Grad-CAM++ and Integrated Gradients"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Detection Model Explanations (Grad-CAM++ & Integrated Gradients)', fontsize=16)
        
        # Original image with bounding boxes
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image with Detections')
        axes[0, 0].axis('off')
        
        for i, pred in enumerate(explanations['predictions'][:3]):  # Show max 3 detections
            if i >= 3:
                break
                
            x1, y1, x2, y2 = pred['bbox']
            conf = pred['confidence']
            cls = pred['class_id']
            
            # Draw bounding box
            axes[0, 0].add_patch(plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                                            fill=False, color='red', linewidth=2))
            axes[0, 0].text(x1, y1-5, f'Class {cls}: {conf:.2f}', 
                          color='white', fontsize=10, weight='bold',
                          bbox=dict(boxstyle='round', facecolor='red', alpha=0.7))
        
        # Grad-CAM++ heatmap
        if explanations['gradcam_pp_maps'] and explanations['gradcam_pp_maps'][0] is not None:
            axes[0, 1].imshow(explanations['gradcam_pp_maps'][0], cmap='jet')
            axes[0, 1].set_title('Grad-CAM++ Heatmap')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'Grad-CAM++ Not Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].axis('off')
        
        # Integrated Gradients
        if explanations['integrated_gradients_maps'] and explanations['integrated_gradients_maps'][0] is not None:
            axes[1, 0].imshow(explanations['integrated_gradients_maps'][0], cmap='RdBu_r')
            axes[1, 0].set_title('Integrated Gradients')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'Integrated Gradients Not Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # Summary statistics
        confidences = [pred['confidence'] for pred in explanations['predictions']]
        if confidences:
            axes[1, 1].text(0.1, 0.9, f'Total Detections: {len(explanations["predictions"])}', 
                           transform=axes[1, 1].transAxes, fontsize=12, weight='bold')
            axes[1, 1].text(0.1, 0.75, f'Avg Confidence: {np.mean(confidences):.3f}', 
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.6, f'Min Confidence: {np.min(confidences):.3f}', 
                           transform=axes[1, 1].transAxes, fontsize=12)
            axes[1, 1].text(0.1, 0.45, f'Max Confidence: {np.max(confidences):.3f}', 
                           transform=axes[1, 1].transAxes, fontsize=12)
            
            # Bar chart of confidences
            ax_bar = axes[1, 1].inset_axes([0.1, 0.05, 0.85, 0.35])
            ax_bar.bar(range(len(confidences[:5])), confidences[:5], color='steelblue')
            ax_bar.set_xlabel('Detection Index')
            ax_bar.set_ylabel('Confidence')
            ax_bar.set_title('Top 5 Detection Confidences', fontsize=10)
            
        axes[1, 1].set_title('Summary Statistics')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'detection' / f'detection_explanation_{Path(explanations["image_path"]).stem}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_classification_explanations(self, image: np.ndarray, explanations: Dict, class_names: List[str]):
        """Create visualization for classification explanations with Grad-CAM++ and Integrated Gradients"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Classification Model Explanations (Grad-CAM++ & Integrated Gradients)', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Prediction info
        pred_class = explanations['predicted_class_name']
        confidence = explanations['confidence']
        axes[0, 0].text(10, 30, f'Predicted: {pred_class}\nConfidence: {confidence:.3f}', 
                        color='white', fontsize=12, weight='bold',
                        bbox=dict(boxstyle='round', facecolor='green', alpha=0.7))
        
        # Grad-CAM++
        if explanations['gradcam_pp_map'] is not None:
            axes[0, 1].imshow(explanations['gradcam_pp_map'], cmap='jet')
            axes[0, 1].set_title('Grad-CAM++ Heatmap')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'Grad-CAM++ Not Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].axis('off')
        
        # Integrated Gradients
        if explanations['integrated_gradients'] is not None:
            axes[1, 0].imshow(explanations['integrated_gradients'], cmap='RdBu_r')
            axes[1, 0].set_title('Integrated Gradients')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'Integrated Gradients Not Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # Top 5 predictions and summary
        probs = explanations['all_probabilities']
        top5_indices = np.argsort(probs)[-5:][::-1]
        top5_probs = [probs[i] for i in top5_indices]
        top5_names = [class_names[i] if i < len(class_names) else f'Class_{i}' for i in top5_indices]
        
        axes[1, 1].barh(range(len(top5_names)), top5_probs, color='steelblue')
        axes[1, 1].set_yticks(range(len(top5_names)))
        axes[1, 1].set_yticklabels(top5_names, fontsize=9)
        axes[1, 1].set_title('Top 5 Predictions')
        axes[1, 1].set_xlabel('Probability')
        axes[1, 1].invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'classification' / f'classification_explanation_{Path(explanations["image_path"]).stem}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def _visualize_tampered_detection_explanations(self, image: np.ndarray, explanations: Dict):
        """Create visualization for tampered detection explanations with Grad-CAM++ and Integrated Gradients"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Tampered Detection Model Explanations (Grad-CAM++ & Integrated Gradients)', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Prediction info
        pred_class = explanations['predicted_class_name']
        confidence = explanations['confidence']
        clean_prob = explanations['clean_probability']
        tampered_prob = explanations['tampered_probability']
        
        color = 'green' if pred_class == 'Clean' else 'red'
        axes[0, 0].text(10, 30, f'Predicted: {pred_class}\nConfidence: {confidence:.3f}', 
                        color='white', fontsize=12, weight='bold',
                        bbox=dict(boxstyle='round', facecolor=color, alpha=0.7))
        
        # Grad-CAM++
        if explanations['gradcam_pp_map'] is not None:
            axes[0, 1].imshow(explanations['gradcam_pp_map'], cmap='jet')
            axes[0, 1].set_title('Grad-CAM++ Heatmap')
            axes[0, 1].axis('off')
        else:
            axes[0, 1].text(0.5, 0.5, 'Grad-CAM++ Not Available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].axis('off')
        
        # Integrated Gradients
        if explanations['integrated_gradients'] is not None:
            axes[1, 0].imshow(explanations['integrated_gradients'], cmap='RdBu_r')
            axes[1, 0].set_title('Integrated Gradients')
            axes[1, 0].axis('off')
        else:
            axes[1, 0].text(0.5, 0.5, 'Integrated Gradients Not Available', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].axis('off')
        
        # Probability distribution
        classes = ['Clean', 'Tampered']
        probs = [clean_prob, tampered_prob]
        colors_bar = ['green', 'red']
        
        axes[1, 1].bar(classes, probs, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
        axes[1, 1].set_title('Probability Distribution')
        axes[1, 1].set_ylabel('Probability')
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add value labels on bars
        for i, (cls, prob) in enumerate(zip(classes, probs)):
            axes[1, 1].text(i, prob + 0.02, f'{prob:.3f}', ha='center', va='bottom', 
                           fontsize=11, weight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'tampered_detection' / f'tampered_detection_explanation_{Path(explanations["image_path"]).stem}.png', 
                   dpi=150, bbox_inches='tight')
        plt.close()
    
    def batch_explain(self, model, image_paths: List[str], model_type: str, 
                     class_names: Optional[List[str]] = None) -> Dict:
        """
        Batch explanation for multiple images
        
        Args:
            model: Trained model
            image_paths: List of image paths
            model_type: 'detection' or 'tampered_detection'
            class_names: Class names (not used, kept for compatibility)
            
        Returns:
            Dictionary with all explanations
        """
        print(f"🔍 Batch explaining {len(image_paths)} images for {model_type} model...")
        
        all_explanations = {}
        
        for i, image_path in enumerate(image_paths):
            print(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
            
            try:
                if model_type == 'detection':
                    explanation = self.explain_detection(model, image_path)
                elif model_type == 'tampered_detection' or model_type == 'fake_detection':
                    explanation = self.explain_fake_detection(model, image_path)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                all_explanations[image_path] = explanation
                
            except Exception as e:
                print(f"❌ Error explaining {image_path}: {e}")
                all_explanations[image_path] = {'error': str(e)}
        
        return all_explanations
