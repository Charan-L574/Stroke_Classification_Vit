import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from src.data_preprocessing.biomarker_data_preprocessing import preprocess_biomarker_data
from src.models.biomarker_model import BiomarkerModel
import pandas as pd

def train_biomarker_model():
    """
    Trains the biomarker model.
    """
    # 1. Preprocess the data
    # We'll use the healthcare dataset for initial training
    file_path = 'dataset/healthcare-dataset-stroke-data.csv'
    preprocessor, X_train, X_test, y_train, y_test = preprocess_biomarker_data(file_path)

    # Fit and transform the training data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_processed.toarray() if hasattr(X_train_processed, "toarray") else X_train_processed, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test_processed.toarray() if hasattr(X_test_processed, "toarray") else X_test_processed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)

    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # 2. Instantiate the model
    input_dim = X_train_processed.shape[1]
    model = BiomarkerModel(input_dim=input_dim)

    # 3. Define loss and optimizer
    # Handle severe class imbalance (only ~5% stroke cases)
    class_counts = y_train.value_counts()
    print(f"\nClass distribution - No stroke: {class_counts[0]}, Stroke: {class_counts[1]}")
    
    # Use moderately weighted approach - emphasis on stroke detection but not too aggressive
    # Weight formula: balanced weights scaled down to reduce false positives
    ratio = class_counts[0] / class_counts[1]
    class_weights = torch.tensor([1.0, ratio * 0.5], dtype=torch.float32)  # 50% of full balance
    print(f"Using class weights: {class_weights} (ratio: {ratio:.1f}:1)")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)  # Lower learning rate

    # 4. Training loop
    epochs = 50
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    # 5. Evaluation
    model.eval()
    y_pred_list = []
    y_true_list = []
    y_pred_proba = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            y_pred_list.extend(predicted.cpu().numpy())
            y_true_list.extend(labels.cpu().numpy())
            y_pred_proba.extend(probabilities[:, 1].cpu().numpy())  # Stroke probability

    print("\nClassification Report:")
    print(classification_report(y_true_list, y_pred_list))
    print(f"Accuracy: {accuracy_score(y_true_list, y_pred_list):.4f}")
    
    # Show some example predictions for high-risk patients
    print("\nSample predictions for actual stroke cases:")
    stroke_indices = [i for i, label in enumerate(y_true_list) if label == 1][:10]
    for idx in stroke_indices:
        print(f"  Actual: Stroke, Predicted: {'Stroke' if y_pred_list[idx] == 1 else 'No Stroke'}, Probability: {y_pred_proba[idx]:.2%}")

    # 6. Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'preprocessor': preprocessor,
    }, 'src/prediction/biomarker_model_weights.pth')
    print("\nTrained biomarker model saved to src/prediction/biomarker_model_weights.pth")

if __name__ == '__main__':
    train_biomarker_model()
