import os
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from src.config import PROCESSED_DIR, IMAGE_SIZE, BATCH_SIZE

def get_data_loaders(data_dir=PROCESSED_DIR):
    """
    Creates training and validation DataLoaders with standard augmentations.
    Expects data_dir to have 'train' and 'val' subfolders containing class directories.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")

    # If processed split doesn't exist yet, point to raw or fallback structure
    if not os.path.exists(train_dir):
        train_dir = os.path.join(data_dir, "raw") # fallback if organized manually
        val_dir = train_dir

    train_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, train_dataset.classes