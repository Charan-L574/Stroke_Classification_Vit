import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import os
import numpy as np

from ..models.eeg_model import EEGModel
from ..data_preprocessing.eeg_preprocessing import get_eeg_spectrograms

class EEGDataset(Dataset):
    """Custom PyTorch Dataset for EEG motor imagery data."""
    def __init__(self, bids_root, subject_ids):
        self.bids_root = bids_root
        self.subject_ids = subject_ids
        
        self.all_spectrograms = []
        self.all_labels = []
        
        self._load_data()

    def _load_data(self):
        print("Loading and preprocessing EEG data for all subjects...")
        for subject_id in self.subject_ids:
            spectrograms, labels = get_eeg_spectrograms(self.bids_root, subject_id)
            if spectrograms is not None and labels is not None:
                self.all_spectrograms.append(spectrograms)
                self.all_labels.append(labels)
        
        if not self.all_spectrograms:
            raise RuntimeError("Could not load any EEG data. Please check the dataset path and format.")

        self.all_spectrograms = np.concatenate(self.all_spectrograms, axis=0)
        self.all_labels = np.concatenate(self.all_labels, axis=0)
        
        # Convert to tensors
        self.all_spectrograms = torch.tensor(self.all_spectrograms, dtype=torch.float32)
        self.all_labels = torch.tensor(self.all_labels, dtype=torch.long)

    def __len__(self):
        return len(self.all_labels)

    def __getitem__(self, idx):
        return self.all_spectrograms[idx], self.all_labels[idx]

def train_eeg_model(bids_root='eeg_stroke_patients/edffile/', num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    Trains the EEG model on the motor imagery dataset and saves the weights.
    """
    print("Training EEG Model...")
    
    # Get list of subjects from the directory names
    try:
        subject_dirs = [d for d in os.listdir(bids_root) if d.startswith('sub-') and os.path.isdir(os.path.join(bids_root, d))]
        subject_ids = [d.replace('sub-', '') for d in subject_dirs]
        if not subject_ids:
            raise FileNotFoundError
    except FileNotFoundError:
        print(f"Error: Could not find subject directories in '{bids_root}'.")
        return

    # Create dataset
    full_dataset = EEGDataset(bids_root, subject_ids)
    
    # Split dataset into training and validation sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Get model parameters from the first data sample
    sample_spec, _ = full_dataset[0]
    _, num_channels, num_freq_bins, _ = sample_spec.shape
    num_classes = len(torch.unique(full_dataset.all_labels))

    # Initialize model, loss, and optimizer
    eeg_feature_extractor = EEGModel(num_channels=num_channels, num_freq_bins=num_freq_bins, feature_vector_size=64)
    model = nn.Sequential(
        eeg_feature_extractor,
        nn.Linear(64, num_classes) # Classification head
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (eeg_data, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(eeg_data)
            loss = criterion(outputs, labels)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss / len(train_loader):.4f}')
        
        # Validation loop
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for eeg_data, labels in val_loader:
                outputs = model(eeg_data)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

    # Save the weights of the feature extractor part
    torch.save(eeg_feature_extractor.state_dict(), 'models/eeg_model_weights.pth')
    print("EEG model weights saved to models/eeg_model_weights.pth")

if __name__ == '__main__':
    train_eeg_model()
