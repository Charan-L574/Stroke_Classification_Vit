"""
Enhanced Biomarker Model Training Script
Uses all 3 datasets merged (114,000+ samples)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import sys
from pathlib import Path
import pickle
from datetime import datetime

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.biomarker_model import BiomarkerModel
from src.data_preprocessing.biomarker_data_preprocessing_enhanced import preprocess_biomarker_data_enhanced

def train_biomarker_model_enhanced(
    use_all_datasets=True,
    num_epochs=100,
    batch_size=256,
    learning_rate=0.001,
    save_path='src/prediction/biomarker_model_weights.pth'
):
    """
    Train the biomarker model using enhanced preprocessing (all 3 datasets).
    
    Args:
        use_all_datasets (bool): If True, uses all 3 datasets (114K samples). If False, uses only healthcare dataset (5K samples)
        num_epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for optimizer
        save_path (str): Path to save the trained model weights
    """
    print("="*80)
    print("ENHANCED BIOMARKER MODEL TRAINING")
    if use_all_datasets:
        print("Mode: ENHANCED (All 3 datasets merged - 114K samples)")
    else:
        print("Mode: STANDARD (Healthcare dataset only - 5K samples)")
    print("="*80)
    
    # Check device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥️  Using device: {device}")
    
    # Load and preprocess data
    print("\n📂 Loading and preprocessing biomarker data...")
    preprocessor, X_train, X_test, y_train, y_test = preprocess_biomarker_data_enhanced(
        use_all_datasets=use_all_datasets
    )
    
    # Fit preprocessor and transform data
    print("\n⚙️  Fitting preprocessor and transforming data...")
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)
    
    print(f"\n✅ Data preprocessed successfully!")
    print(f"  Training samples: {len(X_train_processed)}")
    print(f"  Test samples: {len(X_test_processed)}")
    print(f"  Input features: {X_train_processed.shape[1]}")
    print(f"  Stroke cases (train): {sum(y_train)} ({sum(y_train)/len(y_train)*100:.2f}%)")
    print(f"  Stroke cases (test): {sum(y_test)} ({sum(y_test)/len(y_test)*100:.2f}%)")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_processed)
    y_train_tensor = torch.LongTensor(y_train.values)
    X_test_tensor = torch.FloatTensor(X_test_processed)
    y_test_tensor = torch.LongTensor(y_test.values)
    
    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print(f"\n🏗️  Initializing Biomarker model...")
    input_dim = X_train_processed.shape[1]
    model = BiomarkerModel(input_dim=input_dim)
    model = model.to(device)
    
    print(f"\n📋 Model Architecture:")
    print(model)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n📊 Model Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    # Handle severe class imbalance
    # Calculate positive weight for BCEWithLogitsLoss
    num_positive = sum(y_train)
    num_negative = len(y_train) - num_positive
    pos_weight = num_negative / num_positive
    
    print(f"\n⚖️  Class imbalance handling:")
    print(f"  Negative samples: {num_negative} ({num_negative/len(y_train)*100:.2f}%)")
    print(f"  Positive samples: {num_positive} ({num_positive/len(y_train)*100:.2f}%)")
    print(f"  Imbalance ratio: {pos_weight:.2f}:1")
    print(f"  Using pos_weight={pos_weight:.2f} in loss function")
    
    # Loss with class weight
    pos_weight_tensor = torch.tensor([pos_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    print(f"\n🚀 Starting training for {num_epochs} epochs...")
    print("="*80)
    
    best_recall = 0.0
    best_epoch = 0
    train_losses = []
    val_metrics = []
    
    # Prepare path for saving preprocessor separately to avoid torch.load unpickle issues
    preprocessor_path = save_path.replace('.pth', '_preprocessor.pkl')

    # If an old checkpoint exists that may contain a pickled sklearn object, move it aside
    save_path_obj = Path(save_path)
    preprocessor_path_obj = Path(preprocessor_path)
    if save_path_obj.exists():
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        backup_path = save_path_obj.with_name(save_path_obj.stem + f"_backup_{timestamp}" + save_path_obj.suffix)
        save_path_obj.rename(backup_path)
        print(f"Existing checkpoint {save_path} moved to {backup_path}")
    if preprocessor_path_obj.exists():
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        backup_pre = preprocessor_path_obj.with_name(preprocessor_path_obj.stem + f"_backup_{timestamp}" + preprocessor_path_obj.suffix)
        preprocessor_path_obj.rename(backup_pre)
        print(f"Existing preprocessor file {preprocessor_path} moved to {backup_pre}")

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Get logits for second class (stroke)
            logits = outputs[:, 1]
            
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        train_losses.append(epoch_loss)
        
        # Validation phase
        model.eval()
        all_predictions = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                
                # Get probabilities
                probs = torch.softmax(outputs, dim=1)
                stroke_probs = probs[:, 1]
                
                # Predictions (threshold = 0.5)
                predictions = (stroke_probs > 0.5).long()
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(stroke_probs.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        # True Positives, False Positives, True Negatives, False Negatives
        tp = np.sum((all_predictions == 1) & (all_labels == 1))
        fp = np.sum((all_predictions == 1) & (all_labels == 0))
        tn = np.sum((all_predictions == 0) & (all_labels == 0))
        fn = np.sum((all_predictions == 0) & (all_labels == 1))
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        val_metrics.append({
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
        
        # Learning rate scheduling based on recall
        scheduler.step(recall)
        
        # Print progress
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {epoch_loss:.4f}")
        print(f"  Val Metrics - Acc: {accuracy*100:.2f}%, Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1:.3f}")
        print(f"  Confusion Matrix - TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")
        
        # Save best model based on recall (important for medical diagnosis)
        if recall > best_recall:
            best_recall = recall
            best_epoch = epoch + 1
            
            # Save model with preprocessor
            # Save model checkpoint (weights and metadata) WITHOUT the sklearn preprocessor
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'input_dim': input_dim,
                'use_all_datasets': use_all_datasets,
                'epoch': epoch + 1,
                'recall': recall,
                'accuracy': accuracy,
                'f1': f1
            }
            torch.save(checkpoint, save_path)

            # Save the sklearn preprocessor separately using pickle
            try:
                with open(preprocessor_path, 'wb') as f:
                    pickle.dump(preprocessor, f)
            except Exception as e:
                print(f"  ⚠️ Failed to save preprocessor separately: {e}")

            print(f"  ✓ New best model saved! (Recall: {recall*100:.2f}%)")
        
        print("-" * 80)
    
    # Final evaluation
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    
    # Load best model checkpoint (weights and metadata)
    checkpoint = torch.load(save_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # Load the separate sklearn preprocessor
    loaded_preprocessor = None
    try:
        with open(preprocessor_path, 'rb') as f:
            loaded_preprocessor = pickle.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load saved preprocessor from {preprocessor_path}: {e}")
        print("   The model weights were loaded; you may need to recreate or re-fit the preprocessor before using the model in the app.")
    
    print(f"\n🏆 Best Model Performance (Epoch {best_epoch}):")
    print(f"  Recall (Sensitivity): {best_recall*100:.2f}% - Can detect {best_recall*100:.1f}% of strokes")
    
    # Final detailed evaluation
    all_predictions = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            stroke_probs = probs[:, 1]
            predictions = (stroke_probs > 0.5).long()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(stroke_probs.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    # Final confusion matrix
    tp = np.sum((all_predictions == 1) & (all_labels == 1))
    fp = np.sum((all_predictions == 1) & (all_labels == 0))
    tn = np.sum((all_predictions == 0) & (all_labels == 0))
    fn = np.sum((all_predictions == 0) & (all_labels == 1))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n📊 Final Test Set Performance:")
    print(f"  Accuracy: {accuracy*100:.2f}%")
    print(f"  Precision: {precision*100:.2f}% - {precision*100:.1f}% of predicted strokes are correct")
    print(f"  Recall (Sensitivity): {recall*100:.2f}% - Detects {recall*100:.1f}% of actual strokes")
    print(f"  Specificity: {specificity*100:.2f}% - Correctly identifies {specificity*100:.1f}% of non-strokes")
    print(f"  F1 Score: {f1:.3f}")
    
    print(f"\n🔍 Confusion Matrix:")
    print(f"                    Predicted")
    print(f"                 No Stroke  Stroke")
    print(f"  Actual  No:      {tn:6d}   {fp:6d}")
    print(f"          Yes:     {fn:6d}   {tp:6d}")
    
    print(f"\n💾 Model saved to: {save_path}")
    print(f"   Preprocessor saved to: {preprocessor_path}")
    print(f"   Note: preprocessor is saved separately to avoid torch.load unpickle restrictions.")
    
    # Return the model and the loaded preprocessor (if available), otherwise the in-memory preprocessor
    return model, (loaded_preprocessor if loaded_preprocessor is not None else preprocessor), train_losses, val_metrics


if __name__ == '__main__':
    print("\n" + "="*80)
    print("BIOMARKER MODEL TRAINING OPTIONS")
    print("="*80)
    print("\nChoose mode:")
    print("1. STANDARD - Healthcare dataset only (5,109 samples)")
    print("2. ENHANCED - All 3 datasets merged (114,081 samples) ⭐ RECOMMENDED")
    
    choice = input("\nEnter choice (1 or 2, default=2): ").strip() or "2"
    
    use_all = (choice == "2")
    
    # Train the model
    model, preprocessor, train_losses, val_metrics = train_biomarker_model_enhanced(
        use_all_datasets=use_all,
        num_epochs=100,
        batch_size=256,
        learning_rate=0.001,
        save_path='src/prediction/biomarker_model_weights.pth'
    )
    
    print("\n✅ Training complete! Model ready for use in app_advanced.py")
