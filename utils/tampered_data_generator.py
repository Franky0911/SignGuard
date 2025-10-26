"""
Tampered Sign Dataset Generator
Generates tampered traffic signs with various tampering methods:
- Graffiti (random lines, scribbles)
- Overlays (stickers, objects)
- Text overlays
- Blur effects
"""

import os
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
from tqdm import tqdm


class TamperedSignGenerator:
    """Generate tampered traffic signs for training tampered sign detection"""
    
    def __init__(self, output_size=(224, 224)):
        self.output_size = output_size
        self.colors = [
            (0, 0, 0),      # Black
            (255, 255, 255), # White
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Cyan
        ]
    
    def add_graffiti(self, image, intensity='medium'):
        """Add graffiti-style tampering to the image"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Determine number of graffiti elements based on intensity
        if intensity == 'light':
            num_elements = random.randint(1, 3)
        elif intensity == 'medium':
            num_elements = random.randint(3, 6)
        else:  # heavy
            num_elements = random.randint(6, 10)
        
        for _ in range(num_elements):
            color = random.choice(self.colors)
            thickness = random.randint(2, 5)
            
            # Random graffiti type
            graffiti_type = random.choice(['line', 'curve', 'circle', 'scribble'])
            
            if graffiti_type == 'line':
                # Random straight line
                pt1 = (random.randint(0, w), random.randint(0, h))
                pt2 = (random.randint(0, w), random.randint(0, h))
                cv2.line(img, pt1, pt2, color, thickness)
            
            elif graffiti_type == 'curve':
                # Curved line using multiple points
                num_points = random.randint(3, 6)
                points = np.array([[random.randint(0, w), random.randint(0, h)] for _ in range(num_points)])
                cv2.polylines(img, [points], False, color, thickness)
            
            elif graffiti_type == 'circle':
                # Random circle
                center = (random.randint(0, w), random.randint(0, h))
                radius = random.randint(10, min(w, h) // 3)
                cv2.circle(img, center, radius, color, thickness)
            
            else:  # scribble
                # Random scribble
                num_points = random.randint(10, 20)
                points = []
                x, y = random.randint(0, w), random.randint(0, h)
                for _ in range(num_points):
                    x += random.randint(-20, 20)
                    y += random.randint(-20, 20)
                    x = max(0, min(w-1, x))
                    y = max(0, min(h-1, y))
                    points.append([x, y])
                points = np.array(points)
                cv2.polylines(img, [points], False, color, thickness)
        
        return img
    
    def add_overlay(self, image, intensity='medium'):
        """Add overlay tampering (stickers, patches)"""
        img = image.copy()
        h, w = img.shape[:2]
        
        # Determine number of overlays
        if intensity == 'light':
            num_overlays = random.randint(1, 2)
        elif intensity == 'medium':
            num_overlays = random.randint(2, 4)
        else:  # heavy
            num_overlays = random.randint(4, 6)
        
        for _ in range(num_overlays):
            # Random shape
            overlay_type = random.choice(['rectangle', 'circle', 'polygon'])
            color = random.choice(self.colors)
            
            if overlay_type == 'rectangle':
                # Random rectangle overlay
                x1 = random.randint(0, w - 20)
                y1 = random.randint(0, h - 20)
                x2 = random.randint(x1 + 10, min(x1 + w // 2, w))
                y2 = random.randint(y1 + 10, min(y1 + h // 2, h))
                
                # Semi-transparent or solid overlay
                if random.random() > 0.5:
                    # Solid
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                else:
                    # Semi-transparent
                    overlay = img.copy()
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
                    alpha = random.uniform(0.3, 0.7)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            
            elif overlay_type == 'circle':
                # Random circle overlay
                center = (random.randint(0, w), random.randint(0, h))
                radius = random.randint(10, min(w, h) // 3)
                
                if random.random() > 0.5:
                    cv2.circle(img, center, radius, color, -1)
                else:
                    overlay = img.copy()
                    cv2.circle(overlay, center, radius, color, -1)
                    alpha = random.uniform(0.3, 0.7)
                    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)
            
            else:  # polygon
                # Random polygon overlay
                num_points = random.randint(3, 6)
                points = np.array([[random.randint(0, w), random.randint(0, h)] for _ in range(num_points)])
                cv2.fillPoly(img, [points], color)
        
        return img
    
    def add_text(self, image, intensity='medium'):
        """Add text tampering to the image"""
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)
        
        # Determine number of text elements
        if intensity == 'light':
            num_texts = random.randint(1, 2)
        elif intensity == 'medium':
            num_texts = random.randint(2, 3)
        else:  # heavy
            num_texts = random.randint(3, 5)
        
        w, h = img.size
        
        # Common graffiti/vandalism texts
        texts = ['X', 'NO', 'STOP', '!!!', '???', 'OUT', 'FAKE', '666', 'LOL', 'HA', 
                 'BAN', 'FREE', 'LOVE', 'HATE', 'GO', 'YES', 'OFF']
        
        for _ in range(num_texts):
            text = random.choice(texts)
            color = random.choice(self.colors)
            
            # Random position
            x = random.randint(0, max(1, w - 50))
            y = random.randint(0, max(1, h - 30))
            
            # Random font size
            font_size = random.randint(15, 40)
            
            try:
                # Try to use a default font
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                # Fallback to default
                font = ImageFont.load_default()
            
            draw.text((x, y), text, fill=color, font=font)
        
        # Convert back to OpenCV format
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img_cv
    
    def add_blur(self, image, intensity='medium'):
        """Add blur tampering to the image"""
        # Determine blur strength
        if intensity == 'light':
            blur_type = random.choice(['gaussian', 'motion'])
            kernel_size = random.choice([3, 5])
        elif intensity == 'medium':
            blur_type = random.choice(['gaussian', 'motion', 'average'])
            kernel_size = random.choice([5, 7, 9])
        else:  # heavy
            blur_type = random.choice(['gaussian', 'motion', 'average', 'median'])
            kernel_size = random.choice([9, 11, 13, 15])
        
        img = image.copy()
        
        if blur_type == 'gaussian':
            # Gaussian blur
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        
        elif blur_type == 'motion':
            # Motion blur
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
            kernel = kernel / kernel_size
            # Random rotation of motion kernel
            angle = random.randint(0, 180)
            M = cv2.getRotationMatrix2D((kernel_size / 2, kernel_size / 2), angle, 1)
            kernel = cv2.warpAffine(kernel, M, (kernel_size, kernel_size))
            img = cv2.filter2D(img, -1, kernel)
        
        elif blur_type == 'average':
            # Average blur
            img = cv2.blur(img, (kernel_size, kernel_size))
        
        else:  # median
            # Median blur
            img = cv2.medianBlur(img, kernel_size)
        
        return img
    
    def generate_tampered_sign(self, image, tampering_types=None, intensity='medium'):
        """
        Generate a tampered sign with specified tampering types
        
        Args:
            image: Input image (numpy array)
            tampering_types: List of tampering types to apply. 
                           If None, randomly selects 1-3 types
            intensity: Tampering intensity ('light', 'medium', 'heavy')
        
        Returns:
            Tampered image
        """
        if tampering_types is None:
            # Randomly select 1-3 tampering types
            all_types = ['graffiti', 'overlay', 'text', 'blur']
            num_types = random.randint(1, 3)
            tampering_types = random.sample(all_types, num_types)
        
        img = image.copy()
        
        # Apply each tampering type
        for t_type in tampering_types:
            if t_type == 'graffiti':
                img = self.add_graffiti(img, intensity)
            elif t_type == 'overlay':
                img = self.add_overlay(img, intensity)
            elif t_type == 'text':
                img = self.add_text(img, intensity)
            elif t_type == 'blur':
                img = self.add_blur(img, intensity)
        
        return img
    
    def generate_dataset(self, input_dir, output_dir, num_tampered_per_image=3, 
                        test_split=0.2, intensity_distribution=None):
        """
        Generate a complete tampered sign dataset
        
        Args:
            input_dir: Directory containing real traffic sign images
            output_dir: Directory to save tampered dataset
            num_tampered_per_image: Number of tampered versions per real image
            test_split: Ratio of test set
            intensity_distribution: Dict with intensity probabilities
                                   e.g., {'light': 0.3, 'medium': 0.5, 'heavy': 0.2}
        """
        if intensity_distribution is None:
            intensity_distribution = {'light': 0.2, 'medium': 0.6, 'heavy': 0.2}
        
        input_path = Path(input_dir)
        output_path = Path(output_dir)
        
        # Create output directories
        real_train_dir = output_path / 'train' / 'real'
        tampered_train_dir = output_path / 'train' / 'tampered'
        real_test_dir = output_path / 'test' / 'real'
        tampered_test_dir = output_path / 'test' / 'tampered'
        
        for dir_path in [real_train_dir, tampered_train_dir, real_test_dir, tampered_test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Get all image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(input_path.glob(f'*{ext}')))
            image_files.extend(list(input_path.glob(f'*{ext.upper()}')))
        
        if len(image_files) == 0:
            print(f"⚠️ No images found in {input_dir}")
            return
        
        print(f"📁 Found {len(image_files)} images")
        print(f"🔄 Generating {num_tampered_per_image} tampered versions per image")
        
        # Shuffle and split
        random.shuffle(image_files)
        test_count = int(len(image_files) * test_split)
        test_images = image_files[:test_count]
        train_images = image_files[test_count:]
        
        print(f"📊 Train: {len(train_images)}, Test: {len(test_images)}")
        
        # Helper function to select intensity
        def select_intensity():
            r = random.random()
            cumsum = 0
            for intensity, prob in intensity_distribution.items():
                cumsum += prob
                if r <= cumsum:
                    return intensity
            return 'medium'
        
        # Process training images
        print("\n🚀 Processing training images...")
        real_count = 0
        tampered_count = 0
        
        for img_file in tqdm(train_images, desc="Training set"):
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Resize to standard size
            img = cv2.resize(img, self.output_size)
            
            # Save real image
            real_output = real_train_dir / f'real_{real_count:05d}.png'
            cv2.imwrite(str(real_output), img)
            real_count += 1
            
            # Generate tampered versions
            for j in range(num_tampered_per_image):
                intensity = select_intensity()
                tampered_img = self.generate_tampered_sign(img, intensity=intensity)
                
                tampered_output = tampered_train_dir / f'tampered_{tampered_count:05d}.png'
                cv2.imwrite(str(tampered_output), tampered_img)
                tampered_count += 1
        
        # Process test images
        print("\n🔍 Processing test images...")
        real_test_count = 0
        tampered_test_count = 0
        
        for img_file in tqdm(test_images, desc="Test set"):
            # Read image
            img = cv2.imread(str(img_file))
            if img is None:
                continue
            
            # Resize to standard size
            img = cv2.resize(img, self.output_size)
            
            # Save real image
            real_output = real_test_dir / f'real_{real_test_count:05d}.png'
            cv2.imwrite(str(real_output), img)
            real_test_count += 1
            
            # Generate tampered versions
            for j in range(num_tampered_per_image):
                intensity = select_intensity()
                tampered_img = self.generate_tampered_sign(img, intensity=intensity)
                
                tampered_output = tampered_test_dir / f'tampered_{tampered_test_count:05d}.png'
                cv2.imwrite(str(tampered_output), tampered_img)
                tampered_test_count += 1
        
        print("\n✅ Dataset generation complete!")
        print(f"📊 Training set: {real_count} real, {tampered_count} tampered")
        print(f"📊 Test set: {real_test_count} real, {tampered_test_count} tampered")
        print(f"📁 Dataset saved to: {output_path}")
        
        # Save dataset info
        info = {
            'total_images': len(image_files),
            'train_images': len(train_images),
            'test_images': len(test_images),
            'train_real': real_count,
            'train_tampered': tampered_count,
            'test_real': real_test_count,
            'test_tampered': tampered_test_count,
            'num_tampered_per_image': num_tampered_per_image,
            'intensity_distribution': intensity_distribution,
            'output_size': self.output_size
        }
        
        import json
        with open(output_path / 'dataset_info.json', 'w') as f:
            json.dump(info, f, indent=4)
        
        print(f"📝 Dataset info saved to: {output_path / 'dataset_info.json'}")
        
        return info


def main():
    """CLI interface for dataset generation"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate tampered traffic sign dataset')
    parser.add_argument('--input', type=str, required=True,
                       help='Input directory containing real traffic sign images')
    parser.add_argument('--output', type=str, default='dataset/tampered_signs',
                       help='Output directory for tampered dataset')
    parser.add_argument('--num-tampered', type=int, default=3,
                       help='Number of tampered versions per real image')
    parser.add_argument('--test-split', type=float, default=0.2,
                       help='Test set split ratio')
    parser.add_argument('--output-size', type=int, nargs=2, default=[224, 224],
                       help='Output image size (width height)')
    
    args = parser.parse_args()
    
    print("🚀 Tampered Sign Dataset Generator")
    print("=" * 50)
    print(f"Input directory: {args.input}")
    print(f"Output directory: {args.output}")
    print(f"Tampered versions per image: {args.num_tampered}")
    print(f"Test split: {args.test_split}")
    print(f"Output size: {args.output_size}")
    print("=" * 50)
    
    generator = TamperedSignGenerator(output_size=tuple(args.output_size))
    
    generator.generate_dataset(
        input_dir=args.input,
        output_dir=args.output,
        num_tampered_per_image=args.num_tampered,
        test_split=args.test_split
    )


if __name__ == '__main__':
    main()

