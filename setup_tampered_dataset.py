#!/usr/bin/env python3
"""
Helper script to generate tampered sign dataset with correct settings
Prevents the common mistake of using too many source images
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path

def print_banner():
    print("=" * 70)
    print("🎯 Tampered Sign Dataset Setup Helper")
    print("=" * 70)
    print()

def count_images(directory):
    """Count PNG images in a directory"""
    if not os.path.exists(directory):
        return 0
    return len(list(Path(directory).glob("*.png")))

def create_subset(source_dir, output_dir, num_images):
    """Create a subset of images from source directory"""
    print(f"📂 Creating subset from: {source_dir}")
    print(f"   Target: {num_images} images")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get source images
    source_images = list(Path(source_dir).glob("*.png"))
    
    if len(source_images) == 0:
        print("❌ No images found in source directory!")
        return False
    
    if len(source_images) < num_images:
        print(f"⚠️  Only {len(source_images)} images available, using all of them")
        num_images = len(source_images)
    
    # Copy subset
    print(f"   Copying {num_images} images...")
    for i, img in enumerate(source_images[:num_images]):
        if (i + 1) % 100 == 0 or (i + 1) == num_images:
            print(f"   Progress: {i+1}/{num_images}", end='\r')
        shutil.copy2(img, output_dir)
    
    print(f"\n✅ Created subset: {count_images(output_dir)} images")
    return True

def generate_tampered_dataset(input_dir, output_dir, num_tampered=3):
    """Run the tampered data generator"""
    print(f"\n🔧 Generating tampered dataset...")
    print(f"   Input: {input_dir}")
    print(f"   Output: {output_dir}")
    print(f"   Tampered per image: {num_tampered}")
    
    cmd = [
        sys.executable,
        "utils/tampered_data_generator.py",
        "--input", input_dir,
        "--output", output_dir,
        "--num-tampered", str(num_tampered),
        "--test-split", "0.2"
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("✅ Tampered dataset generated successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Dataset generation failed: {e}")
        return False

def verify_dataset(dataset_dir):
    """Verify the generated dataset"""
    import json
    
    info_file = Path(dataset_dir) / "dataset_info.json"
    if not info_file.exists():
        print("⚠️  Warning: dataset_info.json not found")
        return
    
    print(f"\n📊 Dataset Verification:")
    with open(info_file) as f:
        info = json.load(f)
    
    print(f"   Total images: {info.get('total_images', 0):,}")
    print(f"   Train images: {info.get('train_real', 0):,} clean + {info.get('train_tampered', 0):,} tampered")
    print(f"   Test images: {info.get('test_real', 0):,} clean + {info.get('test_tampered', 0):,} tampered")
    
    total_train = info.get('train_real', 0) + info.get('train_tampered', 0)
    batch_size = 32
    batches = total_train // batch_size
    
    print(f"\n⏱️  Training Estimates:")
    print(f"   Batches per epoch: ~{batches}")
    print(f"   Estimated time (RTX 4060): ~{batches * 1.5 / 60:.1f} minutes per epoch")
    print(f"   Total training time (20 epochs): ~{batches * 1.5 * 20 / 60:.1f} minutes")
    
    if total_train > 20000:
        print("\n⚠️  WARNING: Dataset might be too large!")
        print("   Consider regenerating with fewer images (1000-2000)")

def cleanup_subset(subset_dir):
    """Remove the temporary subset directory"""
    if os.path.exists(subset_dir):
        print(f"\n🧹 Cleaning up temporary subset...")
        shutil.rmtree(subset_dir)
        print("✅ Cleanup complete")

def main():
    parser = argparse.ArgumentParser(
        description="Generate tampered sign dataset with optimal settings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick setup with 2000 images (recommended)
  python setup_tampered_dataset.py --quick
  
  # Custom number of images
  python setup_tampered_dataset.py --num-images 1000
  
  # Use existing subset directory
  python setup_tampered_dataset.py --source-dir dataset/my_subset --skip-subset
  
  # Advanced options
  python setup_tampered_dataset.py --num-images 5000 --num-tampered 5 --keep-subset
        """
    )
    
    parser.add_argument('--quick', action='store_true',
                       help='Quick setup with recommended settings (2000 images)')
    parser.add_argument('--num-images', type=int, default=2000,
                       help='Number of source images to use (default: 2000)')
    parser.add_argument('--num-tampered', type=int, default=3,
                       help='Number of tampered variations per image (default: 3)')
    parser.add_argument('--source-dir', type=str, default='dataset/processed/train/images',
                       help='Source directory for clean images')
    parser.add_argument('--output-dir', type=str, default='dataset/tampered_signs',
                       help='Output directory for tampered dataset')
    parser.add_argument('--subset-dir', type=str, default='dataset/tampered_subset',
                       help='Temporary directory for image subset')
    parser.add_argument('--skip-subset', action='store_true',
                       help='Skip subset creation (use source-dir directly)')
    parser.add_argument('--keep-subset', action='store_true',
                       help='Keep temporary subset directory after completion')
    parser.add_argument('--no-verify', action='store_true',
                       help='Skip dataset verification')
    
    args = parser.parse_args()
    
    print_banner()
    
    # Validate source directory
    if not os.path.exists(args.source_dir):
        print(f"❌ Source directory not found: {args.source_dir}")
        print("\nAvailable options:")
        for opt in ['dataset/processed/train/images', 'dataset/augmented/train/images', 'dataset/traffic_sign/Train']:
            if os.path.exists(opt):
                count = count_images(opt)
                print(f"   ✅ {opt} ({count:,} images)")
        return 1
    
    # Check if output already exists
    if os.path.exists(args.output_dir):
        response = input(f"\n⚠️  Output directory already exists: {args.output_dir}\nDelete and regenerate? (yes/no): ")
        if response.lower() in ['yes', 'y']:
            shutil.rmtree(args.output_dir)
            print("✅ Removed existing dataset")
        else:
            print("❌ Cancelled")
            return 1
    
    # Step 1: Create subset (unless skipped)
    if args.skip_subset:
        input_dir = args.source_dir
        print(f"📂 Using source directory directly: {input_dir}")
        num_source = count_images(input_dir)
        print(f"   Found {num_source:,} images")
    else:
        input_dir = args.subset_dir
        if not create_subset(args.source_dir, args.subset_dir, args.num_images):
            return 1
    
    # Step 2: Generate tampered dataset
    if not generate_tampered_dataset(input_dir, args.output_dir, args.num_tampered):
        return 1
    
    # Step 3: Verify dataset
    if not args.no_verify:
        verify_dataset(args.output_dir)
    
    # Step 4: Cleanup
    if not args.skip_subset and not args.keep_subset:
        cleanup_subset(args.subset_dir)
    
    print("\n" + "=" * 70)
    print("✅ Dataset setup complete!")
    print("=" * 70)
    print("\n🚀 Next steps:")
    print("   1. Run: python training.py")
    print("   2. Wait ~10-15 minutes for training to complete")
    print("   3. Check models/tampered_sign_detector.pth")
    print()
    
    return 0

if __name__ == '__main__':
    sys.exit(main())

