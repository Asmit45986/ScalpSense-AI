import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src.config import DEVICE, EPOCHS, LEARNING_RATE, SAVED_MODELS_DIR
from src.dataset import get_data_loaders
from src.model import get_scalp_model

def train_model():
    print(f"Initializing training on device: {DEVICE}")
    train_loader, val_loader, classes = get_data_loaders()
    print(f"Loaded classes: {classes}")

    model = get_scalp_model(num_classes=len(classes)).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        for images, labels in loop:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

            loop.set_postfix(loss=loss.item())

        train_acc = correct_train / total_train

        # Validation phase
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_acc = correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)

        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {running_loss/len(train_loader):.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # Save best model checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(SAVED_MODELS_DIR, "best_scalp_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved Best Model checkpoint with Val Acc: {val_acc:.4f}")
            

if __name__ == "__main__":
    train_model()