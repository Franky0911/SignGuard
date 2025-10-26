"""
Traffic Sign Dataset Preprocessing
Handles the new CSV-based traffic sign dataset with preprocessing and organization
"""

import os
import pandas as pd
import numpy as np
import cv2
import yaml
import shutil
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

class TrafficSignPreprocessor:
    def __init__(self, data_path="dataset/traffic_sign", output_path="dataset/processed"):
        # Handle relative paths correctly regardless of where script is run from
        if not Path(data_path).is_absolute():
            # If running from tools directory, go up one level
            if Path.cwd().name == 'tools':
                self.data_path = Path('..') / data_path
                self.output_path = Path('..') / output_path
            else:
                self.data_path = Path(data_path)
                self.output_path = Path(output_path)
        else:
            self.data_path = Path(data_path)
            self.output_path = Path(output_path)
        self.class_counts = {}
        self.class_weights = {}
        self.class_names = {}
    
    # Data Loading and Analysis
    def load_metadata(self):
        """Load metadata from CSV files"""
        meta_csv = self.data_path / 'Meta.csv'
        train_csv = self.data_path / 'Train.csv'
        test_csv = self.data_path / 'Test.csv'
        
        # Initialize dataframes
        self.train_df = None
        self.test_df = None
        
        # Load metadata
        if meta_csv.exists():
            meta_df = pd.read_csv(meta_csv)
            print(f"Loaded metadata for {len(meta_df)} classes")
            self.class_names = {row['ClassId']: f"class_{row['ClassId']}" for _, row in meta_df.iterrows()}
        else:
            print(f"⚠️  Meta.csv not found at {meta_csv}")
        
        # Load training data
        if train_csv.exists():
            self.train_df = pd.read_csv(train_csv)
            print(f"Loaded {len(self.train_df)} training samples")
        else:
            print(f"⚠️  Train.csv not found at {train_csv}")
        
        # Load test data
        if test_csv.exists():
            self.test_df = pd.read_csv(test_csv)
            print(f"Loaded {len(self.test_df)} test samples")
        else:
            print(f"⚠️  Test.csv not found at {test_csv}")
        
        return self.train_df, self.test_df
    
    def analyze_class_distribution(self):
        """Analyze class distribution from training data"""
        if hasattr(self, 'train_df') and self.train_df is not None:
            all_classes = self.train_df['ClassId'].tolist()
            self.class_counts = Counter(all_classes)
            print(f"Total classes found: {len(self.class_counts)}")
            print(f"Class distribution: {dict(self.class_counts.most_common(10))}")
            return self.class_counts
        else:
            print("No training data found")
            return {}
    
    def compute_class_weights(self):
        """Compute class weights for imbalanced dataset"""
        if not self.class_counts:
            return {}
        
        classes = list(self.class_counts.keys())
        counts = list(self.class_counts.values())
        
        class_weights = compute_class_weight(
            'balanced',
            classes=np.array(classes),
            y=np.repeat(classes, counts)
        )
        
        self.class_weights = dict(zip(classes, class_weights))
        print(f"Class weights computed: {len(self.class_weights)} classes")
        return self.class_weights
    
    def visualize_class_distribution(self):
        """Create visualization of class distribution"""
        if not self.class_counts:
            print("No class distribution data available")
            return
        
        plt.figure(figsize=(15, 8))
        
        # Top 20 classes
        top_classes = dict(self.class_counts.most_common(20))
        classes = list(top_classes.keys())
        counts = list(top_classes.values())
        
        plt.subplot(1, 2, 1)
        plt.bar(range(len(classes)), counts)
        plt.title('Top 20 Classes Distribution')
        plt.xlabel('Class ID')
        plt.ylabel('Count')
        plt.xticks(range(len(classes)), classes, rotation=45)
        
        # Class weights
        plt.subplot(1, 2, 2)
        weights = [self.class_weights.get(c, 1.0) for c in classes]
        plt.bar(range(len(classes)), weights)
        plt.title('Class Weights (Top 20)')
        plt.xlabel('Class ID')
        plt.ylabel('Weight')
        plt.xticks(range(len(classes)), classes, rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_path / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # YOLO Format Processing
    def create_yolo_annotations(self, df, split_name):
        """Create YOLO format annotations from CSV data"""
        images_dir = self.output_path / split_name / 'images'
        labels_dir = self.output_path / split_name / 'labels'
        
        processed_count = 0
        
        for _, row in df.iterrows():
            img_path = self.data_path / row['Path']
            if not img_path.exists():
                continue
            
            # Load image to get dimensions
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            h, w = image.shape[:2]
            
            # Create YOLO annotation
            class_id = int(row['ClassId'])
            
            # If we have ROI information, use it; otherwise use full image
            if 'Roi.X1' in row and 'Roi.Y1' in row and 'Roi.X2' in row and 'Roi.Y2' in row:
                x1, y1, x2, y2 = row['Roi.X1'], row['Roi.Y1'], row['Roi.X2'], row['Roi.Y2']
                # Convert to YOLO format (normalized coordinates)
                x_center = (x1 + x2) / 2 / w
                y_center = (y1 + y2) / 2 / h
                width = (x2 - x1) / w
                height = (y2 - y1) / h
            else:
                # Use full image
                x_center = 0.5
                y_center = 0.5
                width = 1.0
                height = 1.0
            
            # Copy image to output directory
            output_img_path = images_dir / img_path.name
            cv2.imwrite(str(output_img_path), image)
            
            # Create label file
            label_path = labels_dir / f"{img_path.stem}.txt"
            with open(label_path, 'w') as f:
                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
            
            processed_count += 1
            if processed_count % 100 == 0:
                print(f"Processed {processed_count} files for {split_name}...")
        
        print(f"Completed {split_name}: {processed_count} files processed")
        return processed_count
    
    def split_training_data(self):
        """Split training data into train/validation sets"""
        if not hasattr(self, 'train_df') or self.train_df is None:
            print("No training data available")
            return None, None
        
        # Split training data into train/validation (80/20)
        train_data, val_data = train_test_split(
            self.train_df, 
            test_size=0.2, 
            random_state=42, 
            stratify=self.train_df['ClassId']
        )
        
        print(f"Split training data: {len(train_data)} train, {len(val_data)} validation")
        return train_data, val_data
    
    def create_yaml_config(self):
        """Create YAML configuration file for YOLO training"""
        yaml_content = {
            'train': str(self.output_path / 'train' / 'images'),
            'val': str(self.output_path / 'valid' / 'images'),
            'test': str(self.output_path / 'test' / 'images'),
            'nc': len(self.class_counts),
            'names': [self.class_names.get(i, f'class_{i}') for i in sorted(self.class_counts.keys())],
            'class_weights': self.class_weights,
            'class_distribution': dict(self.class_counts)
        }
        
        yaml_path = self.output_path / 'traffic_sign_config.yaml'
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
        
        print(f"Created YAML config: {yaml_path}")
        return yaml_path
    
    # Classification Data Organization
    def organize_classification_data(self):
        """Organize images into class directories for classification training"""
        classification_dir = Path('dataset/classification')
        train_dir = classification_dir / 'train'
        val_dir = classification_dir / 'val'
        
        # Create classification directory structure
        classification_dir.mkdir(exist_ok=True)
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        
        # Use existing train_df if available, otherwise load from CSV
        if hasattr(self, 'train_df') and self.train_df is not None:
            df = self.train_df
            print(f"Using loaded training data: {len(df)} samples")
        else:
            # Load training data
            train_csv = self.data_path / 'Train.csv'
            if not train_csv.exists():
                print("❌ Train.csv not found")
                return None
            
            df = pd.read_csv(train_csv)
            print(f"Loaded {len(df)} training samples")
        
        # Get unique classes
        unique_classes = sorted(df['ClassId'].unique())
        print(f"Found {len(unique_classes)} classes: {unique_classes}")
        
        # Create class directories
        for class_id in unique_classes:
            (train_dir / str(class_id)).mkdir(parents=True, exist_ok=True)
            (val_dir / str(class_id)).mkdir(parents=True, exist_ok=True)
        
        # Split data into train/validation (80/20)
        train_data, val_data = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            stratify=df['ClassId']
        )
        
        print(f"Split: {len(train_data)} train, {len(val_data)} validation")
        
        # Process training data
        train_count = 0
        for _, row in train_data.iterrows():
            img_path = self.data_path / row['Path']
            if img_path.exists():
                class_id = int(row['ClassId'])
                dest_path = train_dir / str(class_id) / img_path.name
                shutil.copy2(img_path, dest_path)
                train_count += 1
        
        # Process validation data
        val_count = 0
        for _, row in val_data.iterrows():
            img_path = self.data_path / row['Path']
            if img_path.exists():
                class_id = int(row['ClassId'])
                dest_path = val_dir / str(class_id) / img_path.name
                shutil.copy2(img_path, dest_path)
                val_count += 1
        
        print(f"✅ Organized classification data:")
        print(f"  - Training images: {train_count}")
        print(f"  - Validation images: {val_count}")
        print(f"  - Classes: {len(unique_classes)}")
        
        return str(classification_dir)
    
    # Main Processing Pipeline
    def run_preprocessing(self):
        """Run preprocessing pipeline (without augmentation)"""
        print("🚀 Starting Traffic Sign Dataset Preprocessing")
        print("=" * 60)
        
        # Step 1: Create output structure
        print("\n1️⃣ Creating output structure...")
        dirs = ['train/images', 'train/labels', 'valid/images', 'valid/labels', 'test/images', 'test/labels']
        for dir_name in dirs:
            (self.output_path / dir_name).mkdir(parents=True, exist_ok=True)
        
        # Step 2: Load and analyze data
        print("\n2️⃣ Loading and analyzing dataset...")
        self.load_metadata()
        self.analyze_class_distribution()
        self.compute_class_weights()
        self.visualize_class_distribution()
        
        # Step 3: Split training data
        print("\n3️⃣ Splitting training data...")
        train_data, val_data = self.split_training_data()
        
        # Step 4: Process all splits to YOLO format
        print("\n4️⃣ Processing data to YOLO format...")
        total_processed = 0
        
        if train_data is not None:
            count = self.create_yolo_annotations(train_data, 'train')
            total_processed += count
        
        if val_data is not None:
            count = self.create_yolo_annotations(val_data, 'valid')
            total_processed += count
        
        if hasattr(self, 'test_df') and self.test_df is not None:
            count = self.create_yolo_annotations(self.test_df, 'test')
            total_processed += count
        
        # Step 5: Create YAML config
        print("\n5️⃣ Creating YAML configuration...")
        self.create_yaml_config()
        
        # Step 6: Organize classification data
        print("\n6️⃣ Organizing classification data...")
        self.organize_classification_data()
        
        print(f"\n🎉 Preprocessing finished!")
        print(f"📊 Statistics:")
        print(f"  - Processed images: {total_processed}")
        print(f"  - Classes: {len(self.class_counts)}")
        print(f"  - Output directory: {self.output_path}")
        print(f"  - Classification data: dataset/classification")
        print(f"\n💡 Next step: Run augmentation with 'python tools/augmentation.py'")
        
        return {
            'processed_count': total_processed,
            'classes': len(self.class_counts),
            'output_path': str(self.output_path)
        }

if __name__ == "__main__":
    preprocessor = TrafficSignPreprocessor()
    results = preprocessor.run_preprocessing()
    print(f"\n✅ Preprocessing completed successfully!")
    print(f"Results: {results}")