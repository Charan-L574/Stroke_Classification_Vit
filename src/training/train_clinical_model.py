import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import joblib

from ..models.clinical_model import ClinicalModel
from ..data_preprocessing.clinical_data_preprocessing import preprocess_clinical_data

def train_clinical_model():
    """
    Loads, preprocesses, and trains the clinical data model using the updated pipeline.
    """
    print("Starting clinical model training...")

    # --- Configuration ---
    DATA_PATH = 'clinical_lab_data/healthcare-dataset-stroke-data.csv'
    MODEL_SAVE_PATH = 'models/clinical_model_weights.pth'
    
    BATCH_SIZE = 64
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    TEST_SIZE = 0.2

    # --- Data Loading and Preprocessing ---
    if not os.path.exists(DATA_PATH):
        print(f"Error: Data file not found at {DATA_PATH}")
        return
        
    print("Loading and preprocessing data with the new pipeline...")
    df = pd.read_csv(DATA_PATH)
    
    # Use the updated preprocessing function which returns features, labels, and the preprocessor
    X_processed, y_processed, preprocessor = preprocess_clinical_data(df.copy(), fit_smote=True)
    
    X_tensor = torch.tensor(X_processed, dtype=torch.float32)
    y_tensor = torch.tensor(y_processed, dtype=torch.long)

    # --- Dataset and DataLoader ---
    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=TEST_SIZE, random_state=42, stratify=y_tensor
    )

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Data prepared: {len(X_train)} training samples, {len(X_test)} test samples.")

    # --- Model Initialization ---
    input_size = X_train.shape[1]
    feature_vector_size = 16 # This must match the expectation in the FusionModel
    model = ClinicalModel(input_size=input_size, hidden_sizes=[64, 32, 16], feature_vector_size=feature_vector_size)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model initialized with input size {input_size} on {device}.")

    # --- Training Setup ---
    criterion = nn.CrossEntropyLoss()
    # We need a temporary classifier head for training this model standalone
    temp_classifier = nn.Linear(feature_vector_size, len(torch.unique(y_tensor))).to(device)
    optimizer = optim.Adam(list(model.parameters()) + list(temp_classifier.parameters()), lr=LEARNING_RATE)

    # --- Training Loop ---
    print("Starting training loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            feature_vectors = model(inputs)
            outputs = temp_classifier(feature_vectors)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
        
        # --- Validation ---
        model.eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                feature_vectors = model(inputs)
                outputs = temp_classifier(feature_vectors)
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())
        
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Accuracy: {accuracy:.4f}, F1: {f1:.4f}")

    # --- Save the Model and Preprocessor ---
    if not os.path.exists('models'):
        os.makedirs('models')
    
    # Save the model state dictionary
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Clinical model re-trained and saved to {MODEL_SAVE_PATH}")
    
    # Save the fitted preprocessor object
    PREPROCESSOR_SAVE_PATH = 'models/clinical_data_preprocessor.joblib'
    joblib.dump(preprocessor, PREPROCESSOR_SAVE_PATH)
    print(f"Clinical data preprocessor saved to {PREPROCESSOR_SAVE_PATH}")

if __name__ == '__main__':
    train_clinical_model()
