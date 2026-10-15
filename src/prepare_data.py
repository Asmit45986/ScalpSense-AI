import os
import shutil
import random
from src.config import RAW_DIR, PROCESSED_DIR

# Define how raw folders map to your 5 target classes
# Adjust the keys on the left if your raw subfolder names differ slightly
CLASS_MAPPING = {
    "Alopecia areata": "Alopecia Areata",
    "PUNCH-AA": "Alopecia Areata", # If you want to merge PUNCH-AA files
    "Androgenetic alopecia": "Androgenetic Alopecia",
    "Psoriasis": "Psoriasis",
    "Seborrheic dermatitis": "Seborrheic Dermatitis",
    "Non-scalp skin": "Normal Scalp",
    # Add other mappings if needed, or group unused classes into Normal Scalp / ignore
}

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def prepare_and_split_data(split_ratio=0.8):
    print("Starting automated dataset preparation and split...")
    
    # Create processed train and val directories
    train_dir = os.path.join(PROCESSED_DIR, "train")
    val_dir = os.path.join(PROCESSED_DIR, "val")
    
    target_classes = set(CLASS_MAPPING.values())
    for c in target_classes:
        os.makedirs(os.path.join(train_dir, c), exist_ok=True)
        os.makedirs(os.path.join(val_dir, c), exist_ok=True)
        
    # Scan raw directory
    if not os.path.exists(RAW_DIR):
        print(f"Error: Raw directory not found at {RAW_DIR}")
        return

    # Process each raw subfolder
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if not file.lower().endswith(VALID_EXTENSIONS):
                continue
                
            # Find which raw folder this file belongs to
            parent_folder = os.path.basename(root)
            if parent_folder not in CLASS_MAPPING:
                continue
                
            target_class = CLASS_MAPPING[parent_folder]
            src_path = os.path.join(root, file)
            
            # Randomly assign to train (80%) or val (20%)
            dest_folder = train_dir if random.random() < split_ratio else val_dir
            dest_path = os.path.join(dest_folder, target_class, file)
            
            # Avoid name collisions if files share names across different raw folders
            if os.path.exists(dest_path):
                base, ext = os.path.splitext(file)
                file = f"{base}_{random.randint(1000, 9999)}{ext}"
                dest_path = os.path.join(dest_folder, target_class, file)
                
            shutil.copy(src_path, dest_path)

    print("Dataset preparation and 80:20 split completed successfully!")
    print(f"Check your processed folder at: {PROCESSED_DIR}")

if __name__ == "__main__":
    prepare_and_split_data()