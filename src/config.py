import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
SAVED_MODELS_DIR = os.path.join(BASE_DIR, "saved_models")

# Ensure directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Class mapping based on your Punch Scalp dataset categories
CLASSES = {
    0: "Alopecia Areata",
    1: "Androgenetic Alopecia",
    2: "Psoriasis",
    3: "Seborrheic Dermatitis",
    4: "Normal / Healthy Scalp"  # Mapped from non-scalp / control samples
}

NUM_CLASSES = len(CLASSES)

# Training Hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Will automatically use your RTX 3050 GPU