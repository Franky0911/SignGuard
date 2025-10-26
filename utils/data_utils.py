"""
DATA UTILITIES - THE MEAL PREP FOR AI TRAINING
==============================================
This module handles loading, preprocessing, and feeding data to AI models during training.

Think of it as a restaurant's meal prep kitchen:
- Raw ingredients (images) come in
- They're cleaned, cut, and prepared (preprocessing)
- They're served in batches to the chef (AI model)

Why is data loading important?
- AI training is like cooking: garbage in = garbage out
- Proper preprocessing makes training faster and more effective
- Batch loading maximizes GPU utilization

Key Concepts:

1. Dataset Classes:
   - Define HOW to load and preprocess each image
   - Like a recipe for preparing ingredients

2. DataLoader:
   - Loads images in batches efficiently
   - Handles shuffling, parallel loading, memory pinning
   - Like having multiple sous chefs working in parallel

3. Transforms:
   - Image preprocessing operations (resize, normalize, augment)
   - Applied on-the-fly during loading
   - Like prep work before cooking

4. Augmentation:
   - Random modifications to training images
   - Helps model generalize better
   - Like practicing with variations of a recipe
"""

import os  # File system operations
import torch  # PyTorch framework
from torch.utils.data import Dataset, DataLoader  # Data loading utilities
import torchvision.transforms as transforms  # Image transformations
from pathlib import Path  # Clean path handling
import cv2  # OpenCV for image loading
import numpy as np  # Numerical operations
from PIL import Image  # Python Imaging Library
import pandas as pd  # For CSV file handling


class AugmentedDataset(Dataset):
    """
    AUGMENTED TRAFFIC SIGN DATASET
    ===============================
    This class loads augmented traffic sign images for classification training.
    
    What is a PyTorch Dataset?
    - A class that knows how to load and preprocess your data
    - Must implement __len__() and __getitem__() methods
    - Like a smart container that serves data piece by piece
    
    Why augmented?
    - Original dataset might have only 100-200 images per class
    - Augmentation creates 1000s of variations (rotated, flipped, etc.)
    - Helps model learn robust features instead of memorizing
    
    File naming convention:
    - Files named like "class_20_001.png"
    - Middle number is the class ID
    - We extract this to know which sign type it is
    
    Think of it as: A smart photo album that knows how to find and prepare photos for AI
    """
    
    def __init__(self, images_path, transform=None):
        """
        Initialize the dataset
        
        Args:
            images_path: Path to folder containing augmented images
            transform: Image transformations to apply (resize, normalize, etc.)
        """
        self.transform = transform  # Store transform pipeline
        self.samples = []  # Will hold (image_path, class_id) pairs
        self.class_to_idx = {}  # Maps class ID to index (for 43 classes, this is identity)
        
        # Initialize class mapping for 43 classes (0-42)
        # GTSRB has 43 traffic sign types
        for i in range(43):
            self.class_to_idx[i] = i
        
        # Load images and infer classes from filenames
        images_path = Path(images_path)
        if images_path.exists():
            # Get all image files
            image_files = list(images_path.glob('*.png')) + list(images_path.glob('*.jpg'))
            
            for img_path in image_files:
                # Try to extract class from filename (assuming format like "class_0_xxx.png")
                filename = img_path.stem
                try:
                    # Extract class ID from filename
                    if '_' in filename:
                        class_id = int(filename.split('_')[1])
                        # Ensure class_id is within valid range (0-42)
                        if class_id < 0 or class_id > 42:
                            continue
                    else:
                        # Fallback: use hash of filename for consistent class assignment
                        class_id = hash(filename) % 43
                    
                    self.samples.append((str(img_path), class_id))
                except (ValueError, IndexError):
                    # Skip files that don't match expected format
                    continue
        
        print(f"Loaded {len(self.samples)} samples from augmented dataset")
        print(f"Classes found: {sorted(self.class_to_idx.keys())}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a dummy image if loading fails
            image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class FakeDetectionDataset(Dataset):
    """Dataset for fake sign detection (binary classification: real vs fake)"""
    
    def __init__(self, real_path, fake_path, transform=None):
        self.transform = transform
        self.samples = []
        
        # Load real images (label 0)
        if os.path.exists(real_path):
            real_images = list(Path(real_path).glob('*.png'))
            for img_path in real_images:
                self.samples.append((str(img_path), 0))
        
        # Load fake images (label 1)
        if os.path.exists(fake_path):
            fake_images = list(Path(fake_path).glob('*.png'))
            for img_path in fake_images:
                self.samples.append((str(img_path), 1))
        
        print(f"Loaded {len(self.samples)} samples for fake detection")
        print(f"  - Real images: {len([s for s in self.samples if s[1] == 0])}")
        print(f"  - Fake images: {len([s for s in self.samples if s[1] == 1])}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a dummy image if loading fails
            image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class TrafficSignDataset(Dataset):
    """Dataset for traffic sign classification using the new CSV-based format"""
    
    def __init__(self, csv_path, data_root, transform=None, is_test=False):
        self.transform = transform
        self.data_root = Path(data_root)
        self.samples = []
        self.classes = set()
        
        # Load CSV data
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            img_path = self.data_root / row['Path']
            if img_path.exists():
                class_id = int(row['ClassId'])
                # Use ClassId directly - GTSRB dataset has classes 0-42
                self.samples.append((str(img_path), class_id))
                self.classes.add(class_id)
        
        print(f"Loaded {len(self.samples)} samples for traffic sign classification")
        print(f"Classes found: {sorted(self.classes)}")
        print(f"Number of classes: {len(self.classes)}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a dummy image if loading fails
            image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


class ClassificationDataset(Dataset):
    """Dataset for traffic sign classification - updated for new structure"""
    
    def __init__(self, data_path, transform=None):
        self.transform = transform
        self.samples = []
        self.class_to_idx = {}
        
        # Check if we have the new traffic sign dataset structure
        traffic_sign_path = Path(data_path) / 'traffic_sign'
        if traffic_sign_path.exists():
            # Use the new traffic sign dataset
            self._load_from_traffic_sign_dataset(traffic_sign_path)
        else:
            # Fallback to old structure
            self._load_from_old_structure(data_path)
        
        print(f"Loaded {len(self.samples)} samples for classification")
        print(f"Classes: {list(self.class_to_idx.keys())}")
    
    def _load_from_traffic_sign_dataset(self, traffic_sign_path):
        """Load data from the new traffic sign dataset structure"""
        train_csv = traffic_sign_path / 'Train.csv'
        if train_csv.exists():
            df = pd.read_csv(train_csv)
            for _, row in df.iterrows():
                img_path = traffic_sign_path / row['Path']
                if img_path.exists():
                    class_id = int(row['ClassId'])
                    if class_id not in self.class_to_idx:
                        self.class_to_idx[class_id] = len(self.class_to_idx)
                    self.samples.append((str(img_path), class_id))
        
        # If no samples found, try loading from class directories
        if not self.samples:
            train_dir = traffic_sign_path / 'Train'
            if train_dir.exists():
                for class_dir in sorted(train_dir.iterdir()):
                    if class_dir.is_dir():
                        class_id = int(class_dir.name)
                        if class_id not in self.class_to_idx:
                            self.class_to_idx[class_id] = len(self.class_to_idx)
                        
                        for img_path in class_dir.glob('*.png'):
                            self.samples.append((str(img_path), class_id))
    
    def _load_from_old_structure(self, data_path):
        """Load data from old directory structure"""
        # Look for train/val structure first
        train_path = Path(data_path) / 'train'
        val_path = Path(data_path) / 'val'
        
        if train_path.exists() and val_path.exists():
            # Load from train/val structure
            self._load_from_train_val_structure(train_path, val_path)
        elif os.path.exists(data_path):
            # Try direct class directories
            class_dirs = [d for d in Path(data_path).iterdir() if d.is_dir()]
            for idx, class_dir in enumerate(class_dirs):
                self.class_to_idx[class_dir.name] = idx
                
                # Load images from this class
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), idx))
        
        # If no samples found, create dummy data
        if not self.samples:
            print("No classification data found, creating dummy dataset...")
            self._create_dummy_dataset()
    
    def _load_from_train_val_structure(self, train_path, val_path):
        """Load data from train/val directory structure"""
        # Load training data
        for class_dir in train_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                
                class_idx = self.class_to_idx[class_name]
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), class_idx))
        
        # Load validation data
        for class_dir in val_path.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.class_to_idx:
                    self.class_to_idx[class_name] = len(self.class_to_idx)
                
                class_idx = self.class_to_idx[class_name]
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((str(img_path), class_idx))
    
    def _create_dummy_dataset(self):
        """Create dummy classification data for demonstration"""
        # Create dummy samples with random classes
        dummy_images = list(Path('dataset/augmented/train/images').glob('*.jpg'))[:100]
        num_classes = 10  # Dummy number of classes
        
        for img_path in dummy_images:
            class_id = np.random.randint(0, num_classes)
            self.samples.append((str(img_path), class_id))
        
        # Create dummy class names
        for i in range(num_classes):
            self.class_to_idx[f'class_{i}'] = i
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Return a dummy image if loading fails
            image = np.zeros((64, 64, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(label, dtype=torch.long)


def get_fake_detection_loaders(real_path, fake_path, batch_size=32, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2):
    """Create data loaders for fake detection"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = FakeDetectionDataset(real_path, fake_path, transform=transform)
    
    if len(dataset) == 0:
        print("⚠️  No data found for fake detection")
        return None, None
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def get_tampered_detection_loaders(real_path, tampered_path, batch_size=32, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2):
    """
    Create data loaders for tampered sign detection with augmentation to prevent overfitting
    
    Args:
        real_path: Path to real/clean traffic signs
        tampered_path: Path to tampered traffic signs
        batch_size: Batch size for training
        num_workers: Number of worker threads
        pin_memory: Whether to use pinned memory
        persistent_workers: Whether to keep workers alive
        prefetch_factor: Data prefetch factor
    
    Returns:
        train_loader, val_loader
    """
    
    # Enhanced training transforms with strong augmentation to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.3),  # Lower probability for traffic signs
        transforms.RandomRotation(degrees=10),  # Moderate rotation
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.15))  # Occlusion simulation
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Try to load from train/test split structure first
    real_train_path = Path(real_path) / 'train' / 'real'
    tampered_train_path = Path(tampered_path) / 'train' / 'tampered'
    real_test_path = Path(real_path) / 'test' / 'real'
    tampered_test_path = Path(tampered_path) / 'test' / 'tampered'
    
    if real_train_path.exists() and tampered_train_path.exists():
        # Use pre-split dataset
        print("📁 Using pre-split dataset structure")
        train_dataset = FakeDetectionDataset(str(real_train_path), str(tampered_train_path), transform=train_transform)
        
        if real_test_path.exists() and tampered_test_path.exists():
            val_dataset = FakeDetectionDataset(str(real_test_path), str(tampered_test_path), transform=val_transform)
        else:
            # Split training data for validation
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset_temp = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            # Create new dataset with validation transform for the split
            val_dataset = val_dataset_temp
    else:
        # Fallback to original structure
        print("📁 Using original dataset structure with train/val split")
        full_dataset = FakeDetectionDataset(real_path, tampered_path, transform=train_transform)
        
        if len(full_dataset) == 0:
            print("⚠️  No data found for tampered detection")
            return None, None
        
        # Split dataset
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset_temp = torch.utils.data.random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create validation dataset with proper transforms
        val_dataset = FakeDetectionDataset(real_path, tampered_path, transform=val_transform)
        val_indices = val_dataset_temp.indices
        val_dataset = torch.utils.data.Subset(val_dataset, val_indices)
    
    if len(train_dataset) == 0:
        print("⚠️  No training data found for tampered detection")
        return None, None
    
    print(f"📊 Tampered detection dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        drop_last=True  # Drop last incomplete batch for stable training
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def get_classification_loaders(data_path, batch_size=32, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2):
    """Create data loaders for classification"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create dataset
    dataset = ClassificationDataset(data_path, transform=transform)
    
    if len(dataset) == 0:
        print("⚠️  No data found for classification")
        return None, None
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def get_augmented_loaders(data_path, batch_size=32, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2):
    """Create data loaders for the augmented dataset with comprehensive augmentation"""
    
    # Enhanced training transforms with strong augmentation to prevent overfitting
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.1, scale=(0.02, 0.33), ratio=(0.3, 3.3))
    ])
    
    # Validation transforms (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets for train and validation
    train_path = Path(data_path) / 'train' / 'images'
    val_path = Path(data_path) / 'valid' / 'images'
    
    if not train_path.exists() or not val_path.exists():
        print("⚠️  Augmented dataset structure not found, falling back to traffic sign dataset")
        return None, None
    
    # Create datasets
    train_dataset = AugmentedDataset(train_path, transform=train_transform)
    val_dataset = AugmentedDataset(val_path, transform=val_transform)
    
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        print("⚠️  No data found in augmented dataset")
        return None, None
    
    print(f"📊 Augmented dataset loaded: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader


def get_traffic_sign_loaders(data_path, batch_size=32, num_workers=2, pin_memory=True, persistent_workers=True, prefetch_factor=2):
    """Create data loaders for the new traffic sign dataset"""
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    traffic_sign_path = Path(data_path) / 'traffic_sign'
    
    # Create training dataset from CSV
    train_csv = traffic_sign_path / 'Train.csv'
    if train_csv.exists():
        train_dataset = TrafficSignDataset(train_csv, traffic_sign_path, transform=transform)
    else:
        print("⚠️  Train.csv not found, using fallback method")
        train_dataset = ClassificationDataset(data_path, transform=transform)
    
    if len(train_dataset) == 0:
        print("⚠️  No training data found")
        return None, None
    
    # Create validation dataset (use test data for validation)
    test_csv = traffic_sign_path / 'Test.csv'
    if test_csv.exists():
        val_dataset = TrafficSignDataset(test_csv, traffic_sign_path, transform=transform, is_test=True)
    else:
        # Split training data for validation
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        prefetch_factor=prefetch_factor if num_workers > 0 else None
    )
    
    return train_loader, val_loader