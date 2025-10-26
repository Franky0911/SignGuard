"""
Organize classification data from new traffic sign dataset to class-based directory structure
"""

import os
import shutil
import pandas as pd
from pathlib import Path
import yaml
from sklearn.model_selection import train_test_split


def organize_traffic_sign_classification_data():
    """Organize images from new traffic sign dataset into class directories for classification training"""
    
    # Paths
    traffic_sign_path = Path('dataset/traffic_sign')
    classification_dir = Path('dataset/classification')
    train_dir = classification_dir / 'train'
    val_dir = classification_dir / 'val'
    
    # Create classification directory structure
    classification_dir.mkdir(exist_ok=True)
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # Load training data
    train_csv = traffic_sign_path / 'Train.csv'
    if not train_csv.exists():
        print("❌ Train.csv not found in dataset/traffic_sign/")
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
        img_path = traffic_sign_path / row['Path']
        if img_path.exists():
            class_id = int(row['ClassId'])
            dest_path = train_dir / str(class_id) / img_path.name
            shutil.copy2(img_path, dest_path)
            train_count += 1
    
    # Process validation data
    val_count = 0
    for _, row in val_data.iterrows():
        img_path = traffic_sign_path / row['Path']
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


def organize_classification_data():
    """Organize images into class directories for classification training - updated for new dataset"""
    
    # Check if we have the new traffic sign dataset
    traffic_sign_path = Path('dataset/traffic_sign')
    if traffic_sign_path.exists():
        print("Using new traffic sign dataset...")
        return organize_traffic_sign_classification_data()
    
    # Fallback to old method
    print("Using old dataset structure...")
    
    # Load class names from config
    config_path = Path('dataset/augmented/augmented_config.yaml')
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        class_names = config['names']
        print(f"Found {len(class_names)} classes")
        
        # Create classification directory structure
        classification_dir = Path('dataset/classification')
        train_dir = classification_dir / 'train'
        val_dir = classification_dir / 'val'
        
        # Create class directories
        for class_name in class_names:
            (train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        # Process training data
        print("Processing training data...")
        train_images_dir = Path('dataset/augmented/train/images')
        train_labels_dir = Path('dataset/augmented/train/labels')
        
        train_count = 0
        for img_path in train_images_dir.glob('*.jpg'):
            # Get corresponding label file
            label_path = train_labels_dir / (img_path.stem + '.txt')
            
            if label_path.exists():
                # Read the first label (assuming single class per image)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        class_id = int(lines[0].split()[0])
                        if 0 <= class_id < len(class_names):
                            class_name = class_names[class_id]
                            
                            # Copy image to class directory
                            dest_path = train_dir / class_name / img_path.name
                            shutil.copy2(img_path, dest_path)
                            train_count += 1
        
        # Process validation data
        print("Processing validation data...")
        val_images_dir = Path('dataset/augmented/valid/images')
        val_labels_dir = Path('dataset/augmented/valid/labels')
        
        val_count = 0
        for img_path in val_images_dir.glob('*.jpg'):
            # Get corresponding label file
            label_path = val_labels_dir / (img_path.stem + '.txt')
            
            if label_path.exists():
                # Read the first label (assuming single class per image)
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        class_id = int(lines[0].split()[0])
                        if 0 <= class_id < len(class_names):
                            class_name = class_names[class_id]
                            
                            # Copy image to class directory
                            dest_path = val_dir / class_name / img_path.name
                            shutil.copy2(img_path, dest_path)
                            val_count += 1
        
        print(f"✅ Organized classification data:")
        print(f"  - Training images: {train_count}")
        print(f"  - Validation images: {val_count}")
        print(f"  - Classes: {len(class_names)}")
        
        return str(classification_dir)
    else:
        print("❌ No dataset found")
        return None


if __name__ == "__main__":
    organize_classification_data()
