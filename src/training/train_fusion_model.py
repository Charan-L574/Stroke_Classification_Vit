import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import os
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
from torch.optim.lr_scheduler import StepLR
import numpy as np
from imblearn.over_sampling import SMOTE
import joblib

from ..data_preprocessing.image_preprocessing import get_image_transforms
from ..data_preprocessing.clinical_data_preprocessing import preprocess_clinical_data
from ..models.fusion_model import FusionModel

class BimodalFusionDataset(Dataset):
    """
    Custom dataset for loading paired image and clinical data.
    It assumes an ordered correspondence between the image files and clinical data rows.
    """
    def __init__(self, image_folder_path, clinical_csv_path, transform=None):
        self.image_dataset = ImageFolder(image_folder_path, transform=transform)
        self.classes = self.image_dataset.classes
        self.class_to_idx = self.image_dataset.class_to_idx
        
        raw_clinical_data = pd.read_csv(clinical_csv_path)
        
        print("Warning: No direct mapping found between image filenames and clinical data IDs.")
        print("Pairing images and clinical rows by their order (i-th image to i-th clinical row).")
        
        # We should not fit SMOTE here to avoid data leakage into the validation set.
        # We will handle class imbalance later using class weights.
        processed_features, processed_labels, self.preprocessor = preprocess_clinical_data(raw_clinical_data.copy(), fit_smote=False)
        
        self.clinical_features = torch.tensor(processed_features, dtype=torch.float32)
        
        # The number of samples is the minimum of the two modalities
        self.num_samples = min(len(self.image_dataset), len(self.clinical_features))
        print(f"Aligned datasets. Using {self.num_samples} samples.")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        image, image_label = self.image_dataset[idx]
        clinical_data_row = self.clinical_features[idx]
        
        # Using the image label as the ground truth.
        return image, clinical_data_row, torch.tensor(image_label, dtype=torch.long)

def train_fusion_model():
    """
    Main function to train the bimodal fusion model.
    """
    print("Starting bimodal fusion model training...")

    # --- Configuration ---
    IMAGE_DATA_PATH = 'MRI_DATA/Stroke_classification'
    CLINICAL_DATA_PATH = 'clinical_lab_data/healthcare-dataset-stroke-data.csv'
    IMAGE_MODEL_WEIGHTS = 'models/image_model_weights.pth'
    CLINICAL_MODEL_WEIGHTS = 'models/clinical_model_weights.pth'
    FUSION_MODEL_SAVE_PATH = 'models/fusion_model_weights.pth'
    
    BATCH_SIZE = 32
    NUM_EPOCHS = 35
    LEARNING_RATE = 1e-5 # Lower learning rate for fine-tuning
    IMAGE_SIZE = 224
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not all(os.path.exists(p) for p in [IMAGE_DATA_PATH, CLINICAL_DATA_PATH, IMAGE_MODEL_WEIGHTS, CLINICAL_MODEL_WEIGHTS]):
        print("Error: Missing necessary data or model weight files.")
        return

    # --- Dataset and DataLoader ---
    print("Loading and pairing datasets...")
    image_transforms = get_image_transforms()
    
    # Create the base ImageFolder dataset to easily get labels for stratification
    # This dataset is only used for splitting, not for training itself.
    base_image_dataset = ImageFolder(IMAGE_DATA_PATH)
    class_names = base_image_dataset.classes
    num_classes = len(class_names)

    # Determine the number of samples to use (minimum of images and clinical data rows)
    raw_clinical_data = pd.read_csv(CLINICAL_DATA_PATH)
    num_samples = min(len(base_image_dataset), len(raw_clinical_data))

    # Create a stratified split on the indices
    indices = list(range(num_samples))
    labels = base_image_dataset.targets[:num_samples] # Use only the labels for the aligned samples
    
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        stratify=labels,
        random_state=42
    )

    # Create full datasets with transforms for train and val
    train_dataset_full = BimodalFusionDataset(
        image_folder_path=IMAGE_DATA_PATH,
        clinical_csv_path=CLINICAL_DATA_PATH,
        transform=image_transforms['train']
    )
    # For validation, we use the 'val' transform
    val_dataset_full = BimodalFusionDataset(
        image_folder_path=IMAGE_DATA_PATH,
        clinical_csv_path=CLINICAL_DATA_PATH,
        transform=image_transforms['val']
    )

    # Create subsets using the stratified indices
    train_dataset = Subset(train_dataset_full, train_indices)
    val_dataset = Subset(val_dataset_full, val_indices)

    # Calculate class weights for the training subset to handle imbalance
    train_targets = [labels[i] for i in train_indices]
    class_counts = Counter(train_targets)
    total_train_samples = len(train_targets)
    # Smoothing: add a small epsilon to avoid division by zero if a class is not in the training set
    class_weights = [total_train_samples / (num_classes * class_counts.get(i, 1e-9)) for i in range(num_classes)]
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print(f"Class counts in training set: {class_counts}")
    print(f"Calculated class weights: {class_weights_tensor}")

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model Initialization ---
    print("Initializing fusion model...")
    clinical_input_size = train_dataset_full.clinical_features.shape[1]

    model = FusionModel(
        image_model_weights=IMAGE_MODEL_WEIGHTS,
        clinical_model_weights=CLINICAL_MODEL_WEIGHTS,
        clinical_input_size=clinical_input_size,
        num_classes=num_classes
    )
    
    model.to(device)
    print(f"Model loaded on {device}.")

    # --- Training Setup ---
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    # --- Training Loop ---
    print("Starting training loop...")
    
    best_val_accuracy = 0.0
    for epoch in range(NUM_EPOCHS):
        model.train()

        running_loss = 0.0
        correct_predictions = 0
        total_samples_epoch = 0
        
        for images, clinical_data, labels_batch in train_loader:
            images, clinical_data, labels_batch = images.to(device), clinical_data.to(device), labels_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(images, clinical_data)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_samples_epoch += labels_batch.size(0)
            correct_predictions += (predicted == labels_batch).sum().item()

        epoch_loss = running_loss / total_samples_epoch
        epoch_acc = correct_predictions / total_samples_epoch
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, clinical_data, labels_batch in val_loader:
                images, clinical_data, labels_batch = images.to(device), clinical_data.to(device), labels_batch.to(device)
                outputs = model(images, clinical_data)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels_batch.size(0)
                val_correct += (predicted == labels_batch).sum().item()

        val_epoch_loss = val_loss / val_total
        val_epoch_acc = val_correct / val_total

        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f} | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")

        # Step the scheduler
        scheduler.step()

        # Save the best model based on validation accuracy
        if val_epoch_acc > best_val_accuracy:
            best_val_accuracy = val_epoch_acc
            torch.save(model.state_dict(), FUSION_MODEL_SAVE_PATH)
            print(f"Best model saved with accuracy: {best_val_accuracy:.4f}")

    print("Training complete.")

if __name__ == '__main__':
    train_fusion_model()
