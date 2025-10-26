"""
Traffic Sign Recognition - Optimized Custom Augmentation
Streamlined augmentation with YOLO bounding box support
"""

import cv2
import numpy as np
import math
from pathlib import Path
import yaml
import random
import shutil
import matplotlib.pyplot as plt

class TrafficSignAugmenter:
    def __init__(self, data_path="../dataset/processed", output_path="../dataset/augmented"):
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.scenarios = self._create_scenarios()
        self.class_names = {}
    
    def _create_scenarios(self):
        """Create augmentation scenarios with their functions"""
        return {
            'weather': [self._add_weather_effect],
            'lighting': [self._adjust_lighting],
            'blur_noise': [self._add_noise, self._add_blur],
            'night': [self._simulate_night],
            'occlusion': [self._add_occlusion],
            'geometric': [self._rotate_image]
        }
    
    # Bounding Box Utilities
    def _parse_yolo_bbox(self, line, w, h):
        """Parse YOLO bbox line to pixel coordinates"""
        parts = line.strip().split()
        if len(parts) != 5:
            return None
        
        class_id, xc, yc, w_norm, h_norm = int(parts[0]), *map(float, parts[1:])
        xc, yc, w_norm, h_norm = xc * w, yc * h, w_norm * w, h_norm * h
        
        return {
            'class_id': class_id,
            'x1': xc - w_norm/2, 'y1': yc - h_norm/2,
            'x2': xc + w_norm/2, 'y2': yc + h_norm/2,
            'x_center': xc, 'y_center': yc, 'width': w_norm, 'height': h_norm
        }
    
    def _bbox_to_yolo(self, bbox, w, h):
        """Convert bbox to YOLO format"""
        coords = [bbox['x_center']/w, bbox['y_center']/h, bbox['width']/w, bbox['height']/h]
        coords = [max(0, min(1, c)) for c in coords]  # Clamp to [0,1]
        return f"{bbox['class_id']} {coords[0]:.6f} {coords[1]:.6f} {coords[2]:.6f} {coords[3]:.6f}"
    
    def _validate_bbox(self, bbox, w, h):
        """Validate and clamp bbox coordinates"""
        bbox['x1'] = max(0, min(w, bbox['x1']))
        bbox['y1'] = max(0, min(h, bbox['y1']))
        bbox['x2'] = max(0, min(w, bbox['x2']))
        bbox['y2'] = max(0, min(h, bbox['y2']))
        
        if bbox['x2'] <= bbox['x1'] or bbox['y2'] <= bbox['y1']:
            return None
        
        # Recalculate center and dimensions
        bbox['x_center'] = (bbox['x1'] + bbox['x2']) / 2
        bbox['y_center'] = (bbox['y1'] + bbox['y2']) / 2
        bbox['width'] = bbox['x2'] - bbox['x1']
        bbox['height'] = bbox['y2'] - bbox['y1']
        return bbox
    
    def _transform_bbox_rotation(self, bbox, w, h, angle):
        """Transform bbox for rotation"""
        angle_rad = math.radians(angle)
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
        cx, cy = w/2, h/2
        
        # Rotate corners
        corners = [(bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y1']), 
                  (bbox['x2'], bbox['y2']), (bbox['x1'], bbox['y2'])]
        
        rotated = []
        for x, y in corners:
            x_rel, y_rel = x - cx, y - cy
            x_new = x_rel * cos_a - y_rel * sin_a + cx
            y_new = x_rel * sin_a + y_rel * cos_a + cy
            rotated.append((x_new, y_new))
        
        # New bbox from rotated corners
        x_coords, y_coords = [c[0] for c in rotated], [c[1] for c in rotated]
        new_bbox = {
            'class_id': bbox['class_id'],
            'x1': min(x_coords), 'y1': min(y_coords),
            'x2': max(x_coords), 'y2': max(y_coords)
        }
        
        # Recalculate center and dimensions
        new_bbox['x_center'] = (new_bbox['x1'] + new_bbox['x2']) / 2
        new_bbox['y_center'] = (new_bbox['y1'] + new_bbox['y2']) / 2
        new_bbox['width'] = new_bbox['x2'] - new_bbox['x1']
        new_bbox['height'] = new_bbox['y2'] - new_bbox['y1']
        return new_bbox
    
    # Augmentation Functions (Consolidated)
    def _add_weather_effect(self, image):
        """Add rain, fog, or snow effect"""
        h, w = image.shape[:2]
        effect_type = random.choice(['rain', 'fog', 'snow'])
        
        if effect_type == 'rain':
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            for _ in range(random.randint(50, 100)):
                x, y = random.randint(0, w), random.randint(0, h)
                length, angle = random.randint(10, 30), random.uniform(-10, 10)
                end_x = int(x + length * math.cos(math.radians(angle)))
                end_y = int(y + length * math.sin(math.radians(angle)))
                cv2.line(overlay, (x, y), (end_x, end_y), (200, 200, 200), 1)
            return cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        
        elif effect_type == 'fog':
            fog = np.ones_like(image, dtype=np.uint8) * random.randint(150, 200)
            alpha = random.uniform(0.1, 0.3)
            return cv2.addWeighted(image, 1 - alpha, fog, alpha, 0)
        
        else:  # snow
            snow = np.zeros((h, w), dtype=np.uint8)
            for _ in range(random.randint(100, 200)):
                x, y, size = random.randint(0, w), random.randint(0, h), random.randint(1, 3)
                cv2.circle(snow, (x, y), size, 255, -1)
            snow_3ch = cv2.cvtColor(snow, cv2.COLOR_GRAY2BGR)
            return cv2.addWeighted(image, 0.8, snow_3ch, 0.2, 0)
    
    def _adjust_lighting(self, image):
        """Adjust brightness and contrast"""
        brightness = random.uniform(-50, 50)
        contrast = random.uniform(0.5, 1.5)
        return cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
    
    def _add_noise(self, image):
        """Add Gaussian noise"""
        noise = np.random.normal(0, random.randint(10, 30), image.shape).astype(np.uint8)
        return cv2.add(image, noise)
    
    def _add_blur(self, image):
        """Add blur effect"""
        blur_type = random.choice(['gaussian', 'motion', 'median'])
        if blur_type == 'gaussian':
            k = random.choice([3, 5, 7])
            return cv2.GaussianBlur(image, (k, k), 0)
        elif blur_type == 'motion':
            kernel = np.zeros((9, 9))
            kernel[4, :] = 1
            return cv2.filter2D(image, -1, kernel)
        else:
            return cv2.medianBlur(image, 5)
    
    def _simulate_night(self, image):
        """Simulate night conditions"""
        result = cv2.convertScaleAbs(image, alpha=0.3, beta=-50)
        # Add blue tint
        result[:, :, 0] = np.clip(result[:, :, 0] * 0.8, 0, 255)  # Reduce red
        result[:, :, 1] = np.clip(result[:, :, 1] * 0.9, 0, 255)  # Reduce green
        result[:, :, 2] = np.clip(result[:, :, 2] * 1.1, 0, 255)  # Increase blue
        # Add noise
        noise = np.random.normal(0, 15, result.shape).astype(np.uint8)
        return cv2.add(result, noise)
    
    def _add_occlusion(self, image):
        """Add random occlusion patches"""
        h, w = image.shape[:2]
        result = image.copy()
        for _ in range(random.randint(1, 5)):
            x1, y1 = random.randint(0, w//2), random.randint(0, h//2)
            x2, y2 = x1 + random.randint(20, 60), y1 + random.randint(20, 60)
            color = tuple(random.randint(0, 255) for _ in range(3))
            cv2.rectangle(result, (x1, y1), (x2, y2), color, -1)
        return result
    
    def _rotate_image(self, image, bboxes=None):
        """Rotate image and transform bboxes"""
        angle = random.uniform(-15, 15)
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        result = cv2.warpAffine(image, matrix, (w, h))
        
        if bboxes is not None:
            transformed = []
            for bbox in bboxes:
                if bbox is not None:
                    new_bbox = self._transform_bbox_rotation(bbox, w, h, angle)
                    validated = self._validate_bbox(new_bbox, w, h)
                    if validated is not None:
                        transformed.append(validated)
            return result, transformed
        return result
    
    # Main Augmentation Pipeline
    def _augment_image(self, image, bboxes=None, scenario=None):
        """Apply augmentations to image and bboxes"""
        if scenario is None:
            scenario = random.choice(list(self.scenarios.keys()))
        
        aug_funcs = self.scenarios[scenario]
        num_augs = random.randint(1, min(3, len(aug_funcs)))
        selected_augs = random.sample(aug_funcs, num_augs)
        
        result = image.copy()
        transformed_bboxes = bboxes.copy() if bboxes is not None else None
        
        for aug_func in selected_augs:
            try:
                if aug_func.__name__ == '_rotate_image' and transformed_bboxes is not None:
                    result, transformed_bboxes = aug_func(result, transformed_bboxes)
                else:
                    result = aug_func(result)
            except Exception as e:
                print(f"Augmentation error: {e}")
                continue
        
        return result, transformed_bboxes
    
    def _setup_directories(self):
        """Create output directory structure"""
        dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels', 'test/images', 'test/labels']
        for dir_name in dirs:
            (self.output_path / dir_name).mkdir(parents=True, exist_ok=True)
        print(f"Created augmentation output structure at {self.output_path}")
    
    def _copy_original_data(self):
        """Copy original data to augmented directory"""
        print("Copying original data...")
        for split in ['train', 'valid', 'test']:
            src_img = self.data_path / split / 'images'
            src_lbl = self.data_path / split / 'labels'
            if src_img.exists():
                dst_img = self.output_path / split / 'images'
                dst_lbl = self.output_path / split / 'labels'
                
                for img_file in src_img.glob('*.png'):
                    shutil.copy2(img_file, dst_img / img_file.name)
                if src_lbl.exists():
                    for lbl_file in src_lbl.glob('*.txt'):
                        shutil.copy2(lbl_file, dst_lbl / lbl_file.name)
        print("Original data copied successfully")
    
    def _augment_split(self, split_name, augmentation_factor=3):
        """Augment a single data split"""
        print(f"Augmenting {split_name} split with factor {augmentation_factor}...")
        
        src_img = self.data_path / split_name / 'images'
        src_lbl = self.data_path / split_name / 'labels'
        if not src_img.exists():
            print(f"No images found for {split_name}")
            return 0
        
        dst_img = self.output_path / split_name / 'images'
        dst_lbl = self.output_path / split_name / 'labels'
        
        augmented_count = 0
        scenarios = list(self.scenarios.keys())
        
        for img_file in src_img.glob('*.png'):
            image = cv2.imread(str(img_file))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Load bboxes
            bboxes = []
            lbl_file = src_lbl / f"{img_file.stem}.txt"
            if lbl_file.exists():
                with open(lbl_file, 'r') as f:
                    for line in f:
                        bbox = self._parse_yolo_bbox(line, w, h)
                        if bbox is not None:
                            bboxes.append(bbox)
            
            # Create augmented versions
            for i in range(augmentation_factor):
                scenario = random.choice(scenarios)
                aug_img, aug_bboxes = self._augment_image(image, bboxes, scenario)
                
                new_name = f"{img_file.stem}_aug_{scenario}_{i}{img_file.suffix}"
                cv2.imwrite(str(dst_img / new_name), aug_img)
                
                # Save labels
                if aug_bboxes:
                    with open(dst_lbl / f"{img_file.stem}_aug_{scenario}_{i}.txt", 'w') as f:
                        for bbox in aug_bboxes:
                            f.write(self._bbox_to_yolo(bbox, w, h) + '\n')
                elif lbl_file.exists():
                    shutil.copy2(lbl_file, dst_lbl / f"{img_file.stem}_aug_{scenario}_{i}.txt")
                
                augmented_count += 1
                if augmented_count % 100 == 0:
                    print(f"Augmented {augmented_count} images...")
        
        print(f"Completed {split_name}: {augmented_count} augmented images created")
        return augmented_count
    
    def _create_visualizations(self):
        """Create augmentation and bbox transformation visualizations"""
        print("Creating visualizations...")
        
        # Find sample image with labels
        sample_img, sample_bboxes = None, None
        for img_file in (self.data_path / 'train' / 'images').glob('*.png'):
            lbl_file = self.data_path / 'train' / 'labels' / f"{img_file.stem}.txt"
            if lbl_file.exists():
                sample_img = cv2.imread(str(img_file))
                sample_img = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
                h, w = sample_img.shape[:2]
                
                with open(lbl_file, 'r') as f:
                    sample_bboxes = [self._parse_yolo_bbox(line, w, h) for line in f]
                    sample_bboxes = [b for b in sample_bboxes if b is not None]
                break
        
        if sample_img is None:
            print("No sample images found for visualization")
            return
        
        # Augmentation examples
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()
        
        axes[0].imshow(sample_img)
        axes[0].set_title('Original')
        axes[0].axis('off')
        
        scenarios = list(self.scenarios.keys())
        for i, scenario in enumerate(scenarios[:7]):
            try:
                aug_img, _ = self._augment_image(sample_img, None, scenario)
                axes[i+1].imshow(aug_img)
                axes[i+1].set_title(f'{scenario.title()} Augmentation')
                axes[i+1].axis('off')
            except Exception as e:
                print(f"Visualization error for {scenario}: {e}")
                axes[i+1].text(0.5, 0.5, f'Error in {scenario}', ha='center', va='center')
                axes[i+1].axis('off')
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'augmentation_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Bbox transformation examples
        if sample_bboxes:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            def draw_bboxes(img, bboxes, ax, title):
                ax.imshow(img)
                ax.set_title(title)
                ax.axis('off')
                for bbox in bboxes:
                    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
                    rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
                    ax.add_patch(rect)
                    ax.text(x1, y1-5, f"Class {bbox['class_id']}", color='red', fontsize=8, weight='bold')
            
            draw_bboxes(sample_img, sample_bboxes, axes[0], 'Original with Bounding Boxes')
            
            rot_img, rot_bboxes = self._rotate_image(sample_img, sample_bboxes)
            draw_bboxes(rot_img, rot_bboxes, axes[1], 'Rotated with Transformed Bounding Boxes')
            
            # YOLO format comparison
            axes[2].text(0.1, 0.8, 'YOLO Format Comparison:', fontsize=12, weight='bold')
            axes[2].text(0.1, 0.7, 'Original:', fontsize=10, weight='bold')
            for i, bbox in enumerate(sample_bboxes[:3]):
                yolo_line = self._bbox_to_yolo(bbox, w, h)
                axes[2].text(0.1, 0.6-i*0.1, f"  {yolo_line}", fontsize=8, fontfamily='monospace')
            
            axes[2].text(0.1, 0.2, 'Transformed:', fontsize=10, weight='bold')
            for i, bbox in enumerate(rot_bboxes[:3]):
                yolo_line = self._bbox_to_yolo(bbox, w, h)
                axes[2].text(0.1, 0.1-i*0.1, f"  {yolo_line}", fontsize=8, fontfamily='monospace')
            
            axes[2].set_xlim(0, 1)
            axes[2].set_ylim(0, 1)
            axes[2].axis('off')
            
            plt.tight_layout()
            plt.savefig(self.output_path / 'bbox_transformation_examples.png', dpi=300, bbox_inches='tight')
            plt.show()
    
    def _create_yaml_config(self):
        """Create YAML configuration for augmented dataset"""
        # Try to load class names from metadata
        meta_csv = Path('../dataset/traffic_sign/Meta.csv')
        if meta_csv.exists():
            import pandas as pd
            meta_df = pd.read_csv(meta_csv)
            class_names = [f"class_{i}" for i in sorted(meta_df['ClassId'].unique())]
            num_classes = len(class_names)
        else:
            # Fallback to default
            class_names = [f"class_{i}" for i in range(43)]
            num_classes = 43
        
        yaml_content = {
            'train': str(self.output_path / 'train' / 'images'),
            'val': str(self.output_path / 'valid' / 'images'),
            'test': str(self.output_path / 'test' / 'images'),
            'nc': num_classes,
            'names': class_names,
            'augmentation_info': {
                'scenarios': list(self.scenarios.keys()),
                'augmentation_factor': 3,
                'total_scenarios': len(self.scenarios),
                'bbox_transformations': {
                    'rotation': 'Bounding boxes are transformed for rotation augmentations',
                    'validation': 'All bounding box coordinates are validated and clamped to [0,1] range',
                    'format': 'YOLO format (class_id x_center y_center width height) - all normalized'
                }
            }
        }
        
        with open(self.output_path / 'augmented_config.yaml', 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        print(f"Created augmented YAML config: {self.output_path / 'augmented_config.yaml'}")
    
    def run_augmentation(self, augmentation_factor=3):
        """Run complete augmentation pipeline"""
        print("Starting Traffic Sign Data Augmentation...")
        
        self._setup_directories()
        print(f"Initialized {len(self.scenarios)} augmentation scenarios")
        
        self._copy_original_data()
        
        total_augmented = 0
        for split in ['train', 'valid', 'test']:
            if (self.data_path / split).exists():
                count = self._augment_split(split, augmentation_factor)
                total_augmented += count
        
        self._create_visualizations()
        self._create_yaml_config()
        
        print(f"\nAugmentation completed!")
        print(f"Total augmented images created: {total_augmented}")
        print(f"Output directory: {self.output_path}")
        print(f"Augmentation scenarios: {list(self.scenarios.keys())}")

if __name__ == "__main__":
    augmenter = TrafficSignAugmenter()
    augmenter.run_augmentation(augmentation_factor=3)