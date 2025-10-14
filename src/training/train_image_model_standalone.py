import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.optim.lr_scheduler import StepLR
import numpy as np
from pathlib import Path
import timm
import pandas as pd

from ..data_preprocessing.image_preprocessing import get_image_transforms
from ..utils.plot_charts import plot_and_save_charts

def train_image_model_standalone():
    """
    Main function to train a standalone image classification model.
    """
    print("Starting standalone image model training...")

    # --- Configuration ---
    IMAGE_DATA_PATH = Path('MRI_DATA/Stroke_classification')
    MODEL_SAVE_PATH = Path('src/prediction/image_only_model_weights.pth')
    HISTORY_CSV_PATH = Path('training_history.csv')
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 35
    LEARNING_RATE = 1e-5  # Lower learning rate for fine-tuning
    IMAGE_SIZE = 224
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not IMAGE_DATA_PATH.exists():
        print(f"Error: Image data path not found at {IMAGE_DATA_PATH}")
        return

    # --- Dataset and DataLoader ---
    print("Loading image dataset...")
    image_transforms = get_image_transforms()
    
    full_dataset = ImageFolder(root=IMAGE_DATA_PATH)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    print(f"Found classes: {class_names}")

    # Create a stratified split on the dataset
    indices = list(range(len(full_dataset)))
    labels = full_dataset.targets
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # Create separate datasets for training and validation with respective transforms
    train_dataset = Subset(ImageFolder(root=IMAGE_DATA_PATH, transform=image_transforms['train']), train_indices)
    val_dataset = Subset(ImageFolder(root=IMAGE_DATA_PATH, transform=image_transforms['val']), val_indices)

    # Calculate class weights for the training subset to handle imbalance
    train_targets = [labels[i] for i in train_indices]
    class_counts = Counter(train_targets)
    total_train_samples = len(train_targets)
    class_weights = [total_train_samples / (num_classes * class_counts.get(i, 1e-9)) for i in range(num_classes)]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"Class counts in training set: {class_counts}")
    print(f"Calculated class weights: {class_weights_tensor}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model Initialization ---
    print("Initializing Vision Transformer (ViT) model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
    
    # Unfreeze all layers for fine-tuning
    for param in model.parameters():
        param.requires_grad = True
        
    model.to(device)
    print(f"Model loaded on {device} for fine-tuning.")

    # --- Training Setup ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    # --- Training Loop ---
    print("Starting training loop...")
    
    best_val_accuracy = 0.0
    training_history = [] # To store metrics for each epoch

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        
        for images, labels_batch in train_loader:
            images, labels_batch = images.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == labels_batch).sum().item()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_predictions / len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        with torch.no_grad():
            for images, labels_batch in val_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_correct += (predicted == labels_batch).sum().item()

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_correct / len(val_dataset)

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

        # Store metrics
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': epoch_loss,
            'train_acc': epoch_acc,
            'val_loss': val_epoch_loss,
            'val_acc': val_epoch_acc
        })

        scheduler.step()

        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Best model saved with accuracy: {best_val_accuracy:.4f}")

    print(f"Training complete. Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # --- Save History and Plot Charts ---
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(HISTORY_CSV_PATH, index=False)
    print(f"Training history saved to {HISTORY_CSV_PATH}")

    print("Generating training charts...")
    plot_and_save_charts(HISTORY_CSV_PATH, save_dir='.', chart_prefix='training_charts')
    print("Charts generated successfully.")

if __name__ == '__main__':
    train_image_model_standalone()
