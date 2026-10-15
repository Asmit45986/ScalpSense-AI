import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.config import DEVICE, SAVED_MODELS_DIR, PROCESSED_DIR
from src.dataset import get_data_loaders
from src.model import get_scalp_model

def evaluate_model():
    _, val_loader, classes = get_data_loaders()
    model_path = os.path.join(SAVED_MODELS_DIR, "best_scalp_model.pth")
    
    if not os.path.exists(model_path):
        print(f"Error: Model checkpoint not found at {model_path}. Train the model first!")
        return

    model = get_scalp_model(num_classes=len(classes)).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    all_preds = []
    all_labels = []

    print("Running evaluation on validation set...")
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))

    # Confusion Matrix Plot
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix - Scalp Disorder Classification')
    
    os.makedirs("outputs/confusion matrix", exist_ok=True)
    plt.savefig("outputs/confusion matrix/confusion_matrix.png")
    print("Saved confusion matrix plot to outputs/confusion matrix/confusion_matrix.png")

if __name__ == "__main__":
    evaluate_model()