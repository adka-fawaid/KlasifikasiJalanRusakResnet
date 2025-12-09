"""
Generate augmented data once and save to disk
"""

import os
from pathlib import Path
from PIL import Image
import torch
import torchvision.transforms as T
from tqdm import tqdm
import shutil

def create_augmented_dataset(processed_dir, augmented_dir, multiplier=5, class_names=None):
    """
    Create augmented dataset from processed train data
    
    Args:
        processed_dir: Path to Data_processed directory
        augmented_dir: Path to Data_augmented directory
        multiplier: How many augmented versions per original image
        class_names: List of class folder names
    """
    
    processed_path = Path(processed_dir)
    augmented_path = Path(augmented_dir)
    
    train_dir = processed_path / "train"
    
    if not train_dir.exists():
        print(f"❌ Train directory not found: {train_dir}")
        return False
    
    print(f"🎨 Creating augmented dataset...")
    print(f"📂 Source: {train_dir}")
    print(f"📂 Target: {augmented_path}")
    print(f"🔢 Multiplier: {multiplier}x")
    
    # Create augmented directory structure
    augmented_path.mkdir(parents=True, exist_ok=True)
    
    # Create train, val, test directories in augmented
    for split in ["train", "val", "test"]:
        split_dir = augmented_path / split
        split_dir.mkdir(exist_ok=True)
        
        if class_names:
            for class_name in class_names:
                (split_dir / class_name).mkdir(exist_ok=True)
    
    # Define augmentation transforms with ColorJitter
    augment_transform = T.Compose([
        T.RandomRotation(15),
        T.RandomHorizontalFlip(0.5),
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    ])
    
    # Copy val and test directories as-is (no augmentation)
    print("\n📋 Copying val and test data (no augmentation)...")
    for split in ["val", "test"]:
        source_split = processed_path / split
        target_split = augmented_path / split
        
        if source_split.exists():
            # Remove existing and copy fresh
            if target_split.exists():
                shutil.rmtree(target_split)
            shutil.copytree(source_split, target_split)
            print(f"✅ Copied {split}: {len(list(source_split.rglob('*.jpg')))} images")
    
    # Process train data with augmentation
    print(f"\n🎨 Generating augmented train data...")
    
    total_original = 0
    total_augmented = 0
    
    for class_name in class_names:
        class_source = train_dir / class_name
        class_target = augmented_path / "train" / class_name
        
        if not class_source.exists():
            print(f"⚠️ Class directory not found: {class_source}")
            continue
        
        # Get all image files
        image_files = list(class_source.glob("*.jpg")) + list(class_source.glob("*.png"))
        
        if not image_files:
            print(f"⚠️ No images found in {class_source}")
            continue
        
        print(f"\n📁 Processing class: {class_name}")
        print(f"   Original images: {len(image_files)}")
        
        class_target.mkdir(parents=True, exist_ok=True)
        
        # Copy original images
        for img_file in tqdm(image_files, desc=f"Copying {class_name}", leave=False):
            target_file = class_target / f"orig_{img_file.name}"
            shutil.copy2(img_file, target_file)
            total_original += 1
        
        # Generate augmented images
        for img_file in tqdm(image_files, desc=f"Augmenting {class_name}", leave=False):
            try:
                # Load image
                img = Image.open(img_file).convert("RGB")
                
                # Generate augmented versions
                for i in range(multiplier):
                    # Apply augmentation
                    aug_img = augment_transform(img)
                    
                    # Save augmented image
                    base_name = img_file.stem
                    aug_filename = f"aug_{i+1}_{base_name}.jpg"
                    aug_path = class_target / aug_filename
                    
                    aug_img.save(aug_path, quality=95)
                    total_augmented += 1
                    
            except Exception as e:
                print(f"⚠️ Error processing {img_file}: {e}")
        
        final_count = len(list(class_target.glob("*.jpg")))
        print(f"   Final images: {final_count} ({len(image_files)} original + {final_count - len(image_files)} augmented)")
    
    print(f"\n✅ Augmented dataset creation completed!")
    print(f"📊 Summary:")
    print(f"   Original train images: {total_original}")
    print(f"   Augmented images: {total_augmented}")
    print(f"   Total train images: {total_original + total_augmented}")
    print(f"📂 Saved to: {augmented_path}")
    
    return True

def check_augmented_dataset_exists(augmented_dir, class_names):
    """Check if augmented dataset already exists and is complete"""
    augmented_path = Path(augmented_dir)
    
    if not augmented_path.exists():
        return False
    
    # Check if all required directories exist
    required_dirs = []
    for split in ["train", "val", "test"]:
        for class_name in class_names:
            required_dirs.append(augmented_path / split / class_name)
    
    for dir_path in required_dirs:
        if not dir_path.exists():
            return False
        
        # Check if directory has images
        image_files = list(dir_path.glob("*.jpg")) + list(dir_path.glob("*.png"))
        if len(image_files) == 0:
            return False
    
    return True

if __name__ == "__main__":
    # Test the function
    processed_dir = "Data_processed"
    augmented_dir = "Data_augmented"
    class_names = ["Jalan Kategori Baik", "Jalan Kurang Baik", "Jalan Rusak"]
    
    create_augmented_dataset(processed_dir, augmented_dir, multiplier=3, class_names=class_names)