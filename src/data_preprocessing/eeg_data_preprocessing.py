import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import ast

def preprocess_eeg_data(file_path='dataset/5 Essential Words For Post-Stroke Patient EEG Dataset.csv'):
    """
    Loads and preprocesses the EEG data from the '5 Essential Words' dataset.
    
    Dataset structure:
    - Column 0: EEG signal array (256 values as string)
    - Column 1: Label metadata array [0, val1, val2, ...] - 5 unique patterns = 5 classes
    - Column 2: Additional metadata
    
    The 5 classes represent different brain states or spoken words:
    - Class 0: [0, 22, 1, 72, 0, 0, 0, 0]  (280 samples - 4.0%)
    - Class 1: [0, 22, 50, 38, 0, 0, 0, 0] (2000 samples - 28.6%)
    - Class 2: [0, 43, 0, 0, 0, 0, 0, 0]   (720 samples - 10.3%)
    - Class 3: [0, 45, 37, 2, 71, 0, 0, 0] (2000 samples - 28.6%)
    - Class 4: [0, 48, 36, 75, 0, 0, 0, 0] (2000 samples - 28.6%)

    Args:
        file_path (str): The path to the EEG CSV file.

    Returns:
        X_train, X_test, y_train, y_test: The split and preprocessed data.
            - X shape: (num_samples, 1, 256) - EEG signals
            - y shape: (num_samples,) - Class labels 0-4
    """
    # Load the CSV file
    df = pd.read_csv(file_path, header=None)
    
    print(f"Loaded {len(df)} EEG samples from {file_path}")
    
    # Parse EEG signals from column 0
    eeg_signals = []
    for signal_str in df[0]:
        # The signal is stored as a string like "[ 3  3  2  1 ...]"
        # Extract all numbers using regex
        import re
        numbers = re.findall(r'-?\d+', signal_str)
        signal_arr = [int(x) for x in numbers]
        eeg_signals.append(signal_arr)
    
    eeg_signals = np.array(eeg_signals, dtype=np.float32)
    
    print(f"Original signal shape: {eeg_signals.shape}")
    
    # The dataset has 512 samples, but our model expects 256
    # We'll downsample by taking every other sample
    if eeg_signals.shape[1] == 512:
        print("Downsampling from 512 to 256 samples...")
        eeg_signals = eeg_signals[:, ::2]  # Take every 2nd sample
    elif eeg_signals.shape[1] != 256:
        # If different length, truncate or pad to 256
        if eeg_signals.shape[1] > 256:
            eeg_signals = eeg_signals[:, :256]
        else:
            pad_width = ((0, 0), (0, 256 - eeg_signals.shape[1]))
            eeg_signals = np.pad(eeg_signals, pad_width, mode='constant')
    
    print(f"Final signal shape: {eeg_signals.shape}")
    
    # Reshape to (num_samples, 1, 256) for CNN input
    X = eeg_signals.reshape(-1, 1, 256)
    
    # Parse labels from column 1
    # Create a mapping from unique label patterns to class indices
    unique_labels = sorted(df[1].unique())
    label_to_class = {label: idx for idx, label in enumerate(unique_labels)}
    
    print(f"\nFound {len(unique_labels)} unique classes:")
    for label, class_idx in label_to_class.items():
        count = (df[1] == label).sum()
        print(f"  Class {class_idx}: {label} ({count} samples)")
    
    # Convert label strings to class indices
    y = df[1].map(label_to_class).values
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y  # Maintain class distribution in splits
    )
    
    print(f"\nTrain set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"\nClass distribution in training set:")
    unique, counts = np.unique(y_train, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  Class {cls}: {cnt} samples ({cnt/len(y_train)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test

if __name__ == '__main__':
    # Example of how to use the function
    file_path = 'dataset/5 Essential Words For Post-Stroke Patient EEG Dataset.csv'
    X_train, X_test, y_train, y_test = preprocess_eeg_data(file_path)
    
    print("Shape of training data:", X_train.shape)
    print("Shape of testing data:", X_test.shape)
    print("Shape of training labels:", y_train.shape)
    print("Shape of testing labels:", y_test.shape)
    print("\nSample label distribution in training set:")
    print(pd.Series(y_train).value_counts())
