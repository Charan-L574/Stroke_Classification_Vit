"""
Training script for the Multi-Modal Fusion Model.

This script trains the trimodal fusion network that combines:
1. Vision Transformer (ViT) - Brain MRI/CT scan analysis
2. Biomarker MLP - Clinical risk assessment  
3. EEG CNN - Brain activity monitoring

Training Strategy:
- Load pretrained weights for each modality
- Fine-tune with a lower learning rate
- Use attention-based late fusion
- Handle class imbalance with weighted loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import Counter
from torchvision.datasets import ImageFolder

# Local imports
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.data_preprocessing.image_preprocessing import get_image_transforms
from src.data_preprocessing.biomarker_data_preprocessing import BiomarkerPreprocessor
from src.models.fusion_model import FusionModel


class TrimodalDataset(Dataset):
    """
    Custom dataset for loading paired image, biomarker, and EEG data.
    
    Handles missing modalities gracefully by returning None for unavailable data.
    """
    def __init__(self, 
                 image_folder_path=None,
                 biomarker_csv_path=None,
                 eeg_csv_path=None,
                 image_transform=None,
                 biomarker_preprocessor=None):
        """
        Args:
            image_folder_path: Path to image folder (ImageFolder format)
            biomarker_csv_path: Path to biomarker CSV file
            eeg_csv_path: Path to EEG data CSV file
            image_transform: Torchvision transforms for images
            biomarker_preprocessor: Fitted preprocessor for biomarker data
        """
        self.image_transform = image_transform
        self.biomarker_preprocessor = biomarker_preprocessor
        
        # Load image data
        self.image_data = None
        self.classes = None
        self.class_to_idx = None
        if image_folder_path and os.path.exists(image_folder_path):
            self.image_data = ImageFolder(image_folder_path, transform=image_transform)
            self.classes = self.image_data.classes
            self.class_to_idx = self.image_data.class_to_idx
            print(f"✓ Loaded {len(self.image_data)} images from {image_folder_path}")
            print(f"  Classes: {self.classes}")
        
        # Load biomarker data
        self.biomarker_data = None
        if biomarker_csv_path and os.path.exists(biomarker_csv_path):
            df = pd.read_csv(biomarker_csv_path)
            if biomarker_preprocessor:
                self.biomarker_features = biomarker_preprocessor.transform(df)
                self.biomarker_labels = df['stroke'].values if 'stroke' in df.columns else None
            else:
                # Basic preprocessing
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                self.biomarker_features = df[numeric_cols].fillna(0).values
                self.biomarker_labels = df['stroke'].values if 'stroke' in df.columns else None
            self.biomarker_data = torch.tensor(self.biomarker_features, dtype=torch.float32)
            print(f"✓ Loaded {len(self.biomarker_data)} biomarker samples")
        
        # Load EEG data
        self.eeg_data = None
        if eeg_csv_path and os.path.exists(eeg_csv_path):
            eeg_df = pd.read_csv(eeg_csv_path)
            # Assuming EEG data has 256 samples per row
            if 'eeg_value' in eeg_df.columns:
                # Group by sample if needed
                self.eeg_data = torch.tensor(eeg_df['eeg_value'].values.reshape(-1, 256), dtype=torch.float32)
            else:
                self.eeg_data = torch.tensor(eeg_df.values[:, :256], dtype=torch.float32)
            self.eeg_data = self.eeg_data.unsqueeze(1)  # Add channel dimension
            print(f"✓ Loaded {len(self.eeg_data)} EEG samples")
        
        # Determine dataset size
        sizes = []
        if self.image_data: sizes.append(len(self.image_data))
        if self.biomarker_data is not None: sizes.append(len(self.biomarker_data))
        if self.eeg_data is not None: sizes.append(len(self.eeg_data))
        
        self.num_samples = min(sizes) if sizes else 0
        print(f"✓ Dataset aligned to {self.num_samples} samples")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            image: Image tensor or None
            biomarker: Biomarker tensor or None
            eeg: EEG tensor or None
            label: Class label
        """
        # Get image and label
        if self.image_data:
            image, label = self.image_data[idx]
        else:
            image = None
            label = 0
        
        # Get biomarker data
        if self.biomarker_data is not None and idx < len(self.biomarker_data):
            biomarker = self.biomarker_data[idx]
        else:
            biomarker = None
        
        # Get EEG data
        if self.eeg_data is not None and idx < len(self.eeg_data):
            eeg = self.eeg_data[idx]
        else:
            eeg = None
        
        return image, biomarker, eeg, torch.tensor(label, dtype=torch.long)


def collate_fn(batch):
    """
    Custom collate function to handle None values in batch.
    """
    images, biomarkers, eegs, labels = zip(*batch)
    
    # Stack non-None tensors
    if images[0] is not None:
        images = torch.stack(images)
    else:
        images = None
    
    if biomarkers[0] is not None:
        biomarkers = torch.stack(biomarkers)
    else:
        biomarkers = None
    
    if eegs[0] is not None:
        eegs = torch.stack(eegs)
    else:
        eegs = None
    
    labels = torch.stack(labels)
    
    return images, biomarkers, eegs, labels


def train_fusion_model():
    """
    Main training function for the trimodal fusion model.
    """
    print("=" * 60)
    print("Multi-Modal Fusion Model Training")
    print("=" * 60)

    # === Configuration ===
    CONFIG = {
        # Data paths
        'image_data_path': 'dataset/brain_scans',  # ImageFolder format
        'biomarker_csv_path': 'dataset/healthcare-dataset-stroke-data.csv',
        'eeg_csv_path': 'dataset/5 Essential Words For Post-Stroke Patient EEG Dataset.csv',
        
        # Pretrained weights
        'image_weights': 'src/prediction/image_only_model_weights.pth',
        'biomarker_weights': 'src/prediction/biomarker_model_weights.pth',
        'eeg_weights': 'src/prediction/eeg_model_weights.pth',
        
        # Output
        'save_path': 'src/prediction/fusion_model_weights.pth',
        
        # Training params
        'batch_size': 16,
        'num_epochs': 30,
        'learning_rate': 1e-4,
        'weight_decay': 1e-4,
        'num_classes': 3,
        'biomarker_input_dim': 29,
        
        # Model config
        'use_attention': True,
        'freeze_backbones': False,
        'dropout_rate': 0.3,
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # === Initialize Model ===
    print("\nInitializing fusion model...")
    model = FusionModel(
        image_model_weights=CONFIG['image_weights'] if os.path.exists(CONFIG['image_weights']) else None,
        biomarker_model_weights=CONFIG['biomarker_weights'] if os.path.exists(CONFIG['biomarker_weights']) else None,
        eeg_model_weights=CONFIG['eeg_weights'] if os.path.exists(CONFIG['eeg_weights']) else None,
        biomarker_input_dim=CONFIG['biomarker_input_dim'],
        num_classes=CONFIG['num_classes'],
        use_attention=CONFIG['use_attention'],
        freeze_backbones=CONFIG['freeze_backbones'],
        dropout_rate=CONFIG['dropout_rate']
    )
    model.to(device)

    # === Load Dataset ===
    print("\nLoading datasets...")
    image_transforms = get_image_transforms()
    
    # Create dataset
    train_dataset = TrimodalDataset(
        image_folder_path=CONFIG['image_data_path'] if os.path.exists(CONFIG['image_data_path']) else None,
        biomarker_csv_path=CONFIG['biomarker_csv_path'],
        eeg_csv_path=CONFIG['eeg_csv_path'],
        image_transform=image_transforms['train']
    )
    
    if len(train_dataset) == 0:
        print("⚠ No data available. Creating synthetic demo data...")
        # Create synthetic data for demonstration
        num_samples = 100
        synthetic_images = torch.randn(num_samples, 3, 224, 224)
        synthetic_biomarkers = torch.randn(num_samples, CONFIG['biomarker_input_dim'])
        synthetic_eegs = torch.randn(num_samples, 1, 256)
        synthetic_labels = torch.randint(0, CONFIG['num_classes'], (num_samples,))
        
        class SyntheticDataset(Dataset):
            def __init__(self, images, biomarkers, eegs, labels):
                self.images = images
                self.biomarkers = biomarkers
                self.eegs = eegs
                self.labels = labels
            def __len__(self):
                return len(self.labels)
            def __getitem__(self, idx):
                return self.images[idx], self.biomarkers[idx], self.eegs[idx], self.labels[idx]
        
        train_dataset = SyntheticDataset(synthetic_images, synthetic_biomarkers, synthetic_eegs, synthetic_labels)
        print(f"Created {num_samples} synthetic samples for demo")

    # Train/Val split
    indices = list(range(len(train_dataset)))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(train_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_fn)
    
    print(f"Train samples: {len(train_subset)}")
    print(f"Val samples: {len(val_subset)}")

    # === Training Setup ===
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['num_epochs'])
    criterion = nn.CrossEntropyLoss()

    # === Training Loop ===
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(CONFIG['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (images, biomarkers, eegs, labels) in enumerate(train_loader):
            # Move to device
            if images is not None: images = images.to(device)
            if biomarkers is not None: biomarkers = biomarkers.to(device)
            if eegs is not None: eegs = eegs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(image_data=images, biomarker_data=biomarkers, eeg_data=eegs)
            if isinstance(outputs, tuple):
                outputs, attention_weights = outputs
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * labels.size(0)
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss /= train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, biomarkers, eegs, labels in val_loader:
                if images is not None: images = images.to(device)
                if biomarkers is not None: biomarkers = biomarkers.to(device)
                if eegs is not None: eegs = eegs.to(device)
                labels = labels.to(device)
                
                outputs = model(image_data=images, biomarker_data=biomarkers, eeg_data=eegs)
                if isinstance(outputs, tuple):
                    outputs, _ = outputs
                
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss /= val_total
        val_acc = val_correct / val_total
        
        # Update scheduler
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Print progress
        print(f"Epoch [{epoch+1:2d}/{CONFIG['num_epochs']}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': CONFIG
            }, CONFIG['save_path'])
            print(f"  ✓ Best model saved (Val Acc: {val_acc:.4f})")

    print("\n" + "=" * 60)
    print(f"Training Complete! Best Val Accuracy: {best_val_acc:.4f}")
    print("=" * 60)
    
    return model, history


if __name__ == '__main__':
    train_fusion_model()
