import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os

# Assuming the necessary modules are in the parent directory
from ..models.image_model import create_image_model
from ..data_preprocessing.image_preprocessing import get_image_transforms

def train_image_model(data_dir='MRI_DATA/Dataset_MRI_Folder/', num_epochs=10, batch_size=32, learning_rate=0.001):
    """
    Trains the image model on the MRI dataset and saves the weights.
    """
    print("Training Image Model...")
    
    # Check if the data directory exists
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return

    # Use torchvision's ImageFolder dataset
    # We need a transform for the images
    image_transforms = get_image_transforms()
    
    try:
        full_dataset = ImageFolder(root=data_dir, transform=image_transforms)
        
        # Split dataset into training and validation sets
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        num_classes = len(full_dataset.classes)
        print(f"Found {num_classes} classes: {full_dataset.class_to_idx}")

    except FileNotFoundError:
        print(f"Could not find the dataset at '{data_dir}'. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the dataset: {e}")
        return

    # Initialize model, loss, and optimizer
    model = create_image_model(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(images)
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
            for images, labels in val_loader:
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        print(f'Validation Accuracy: {100 * correct / total:.2f}%')

    # Save the model weights
    torch.save(model.state_dict(), 'models/image_model_weights.pth')
    print("Image model weights saved to models/image_model_weights.pth")

if __name__ == '__main__':
    train_image_model()
