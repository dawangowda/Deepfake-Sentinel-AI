import os
import shutil
import random

print("Starting dataset setup...")

# --- Configuration ---
SOURCE_DIR = 'Celeb-DF-v2'
OUTPUT_DIR = 'dataset'
TRAIN_SPLIT_RATIO = 0.8

# --- Paths ---
source_real_videos = os.path.join(SOURCE_DIR, 'Celeb-real')
source_fake_videos = os.path.join(SOURCE_DIR, 'Celeb-synthesis')
train_dir = os.path.join(OUTPUT_DIR, 'train')
validation_dir = os.path.join(OUTPUT_DIR, 'validation')
train_real_dir = os.path.join(train_dir, 'real')
train_fake_dir = os.path.join(train_dir, 'fake')
validation_real_dir = os.path.join(validation_dir, 'real')
validation_fake_dir = os.path.join(validation_dir, 'fake')

# --- Create Directories ---
print("Creating target directories...")
os.makedirs(train_real_dir, exist_ok=True)
os.makedirs(train_fake_dir, exist_ok=True)
os.makedirs(validation_real_dir, exist_ok=True)
os.makedirs(validation_fake_dir, exist_ok=True)

# --- Helper Function ---
def split_and_copy_files(source_path, train_dest, val_dest, split_ratio):
    if not os.path.exists(source_path):
        print(f"Warning: Source directory not found at {source_path}")
        return
        
    filenames = os.listdir(source_path)
    random.shuffle(filenames)
    split_point = int(len(filenames) * split_ratio)
    train_files = filenames[:split_point]
    val_files = filenames[split_point:]
    
    print(f"Copying {len(train_files)} files to training set...")
    for filename in train_files:
        shutil.copy(os.path.join(source_path, filename), os.path.join(train_dest, filename))
        
    print(f"Copying {len(val_files)} files to validation set...")
    for filename in val_files:
        shutil.copy(os.path.join(source_path, filename), os.path.join(val_dest, filename))

# --- Process Videos ---
print("\nProcessing REAL videos...")
split_and_copy_files(source_real_videos, train_real_dir, validation_real_dir, TRAIN_SPLIT_RATIO)

print("\nProcessing FAKE videos...")
split_and_copy_files(source_fake_videos, train_fake_dir, validation_fake_dir, TRAIN_SPLIT_RATIO)

print("\nDataset setup complete!")