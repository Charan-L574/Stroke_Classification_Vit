"""
Enhanced EEG Model Training Script
Uses the corrected preprocessing with proper label extraction from CSV
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data_preprocessing.eeg_data_preprocessing import preprocess_eeg_data

# Simple 1D CNN model for EEG classification
class SimpleEEGModel(nn.Module):
    """Simple 1D CNN for EEG signal classification"""
    def __init__(self, num_classes=5, input_length=256):
        super(SimpleEEGModel, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool2 = nn.MaxPool1d(2)
        
        # Calculate flattened size
        # After conv1 + pool1: 256 -> 128
        # After conv2 + pool2: 128 -> 64
        flattened_size = 32 * (input_length // 4)
        
        # Classification layers
        self.fc1 = nn.Linear(flattened_size, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Convolutional layers
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

def train_eeg_model_enhanced(num_epochs=50, batch_size=32, learning_rate=0.001, save_path='src/prediction/eeg_model_weights.pth'):
    """
    Train the EEG model using the properly preprocessed 5 Essential Words dataset.
    
    Args:
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        save_path (str): Path to save the trained model weights
    """
    print("="*80)
    print("ENHANCED EEG MODEL TRAINING")
    print("="*80)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    # Load and preprocess data
    print("\n📂 Loading and preprocessing EEG data...")
    X_train, X_test, y_train, y_test = preprocess_eeg_data()
    
    print(f"\n✅ Data loaded successfully!")
    print(f"  Training samples: {len(X_train)}")
    print(f"  Test samples: {len(X_test)}")
    print(f"  Input shape: {X_train.shape}")
    print(f"  Number of classes: {len(np.unique(y_train))}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print(f"\n🏗️  Initializing EEG model...")
    num_classes = len(np.unique(y_train))
    model = SimpleEEGModel(num_classes=num_classes, input_length=256)
    model = model.to(device)
    
    print(f"\n📋 Model Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Loss and optimizer
    # Handle class imbalance
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    class_weights = torch.FloatTensor(class_weights).to(device)
    
    print(f"\n⚖️  Class weights for imbalanced data:")
    for i, (count, weight) in enumerate(zip(class_counts, class_weights)):
        print(f"  Class {i}: {count} samples, weight={weight:.3f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("="*80)
    
    best_val_acc = 0.0
    best_epoch = 0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        correct_val = 0
        total_val = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_acc = 100 * correct_val / total_val
        val_loss = val_loss / len(test_loader)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {epoch_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
        
        print("-" * 80)
    
    # Final evaluation
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\n🏆 Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    print(f"💾 Model saved to: {save_path}")
    
    # Load best model for final test
    model.load_state_dict(torch.load(save_path, weights_only=True))
    model.eval()
    
    # Detailed per-class metrics
    print("\n📊 Per-Class Performance on Test Set:")
    print("-" * 80)
    
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for label, prediction in zip(labels, predicted):
                if label == prediction:
                    class_correct[label] += 1
                class_total[label] += 1
    
    brain_states = {
        0: "Normal Conscious State",
        1: "Drowsy/Sedated State",
        2: "Deep Sleep/Unconscious",
        3: "Seizure Activity",
        4: "Critical Suppression"
    }
    
    for i in range(num_classes):
        if class_total[i] > 0:
            accuracy = 100 * class_correct[i] / class_total[i]
            state_name = brain_states.get(i, f"Class {i}")
            print(f"Class {i} ({state_name}):")
            print(f"  Accuracy: {accuracy:.2f}% ({class_correct[i]}/{class_total[i]})")
    
    overall_acc = 100 * sum(class_correct) / sum(class_total)
    print(f"\n🎯 Overall Test Accuracy: {overall_acc:.2f}%")
    
    return model, train_losses, val_accuracies


if __name__ == '__main__':
    print("\n" + "="*80)
    print("STARTING ENHANCED EEG MODEL TRAINING")
    print("Dataset: 5 Essential Words For Post-Stroke Patient EEG Dataset")
    print("Total samples: 7,000 (5,600 train / 1,400 test)")
    print("Classes: 5 (brain states)")
    print("="*80)
    
    # Train the model
    model, train_losses, val_accuracies = train_eeg_model_enhanced(
        num_epochs=50,
        batch_size=64,
        learning_rate=0.001,
        save_path='src/prediction/eeg_model_weights.pth'
    )
    
    print("\n✅ Training complete! Model ready for use in app_advanced.py")
