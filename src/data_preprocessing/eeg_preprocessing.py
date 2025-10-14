import numpy as np
from scipy.signal import stft
import mne
from mne_bids import BIDSPath, read_raw_bids

def get_eeg_spectrograms(bids_root, subject_id):
    """
    Loads EEG data for a specific subject from a BIDS-like dataset, extracts epochs 
    for motor imagery, and converts them to spectrograms.

    Args:
        bids_root (str): The root directory of the BIDS dataset (e.g., 'eeg_stroke_patients/edffile/').
        subject_id (str): The subject ID (e.g., '01').

    Returns:
        tuple: A tuple containing:
            - np.ndarray: Spectrograms of shape (num_epochs, num_channels, num_freqs, num_times).
            - np.ndarray: Labels for each epoch.
    """
    # Construct the direct path to the EDF file based on the observed project structure
    raw_path = f"{bids_root}/sub-{subject_id}/eeg/sub-{subject_id}_task-motor-imagery_eeg.edf"
    
    try:
        raw = mne.io.read_raw_edf(raw_path, preload=True, verbose=False)
    except FileNotFoundError:
        print(f"Could not find EEG file for subject {subject_id} at {raw_path}")
        return None, None
    except Exception as e:
        print(f"An error occurred while reading {raw_path} for subject {subject_id}: {e}")
        return None, None

    # Find events from annotations
    try:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
    except ValueError:
        # Some files might not have annotations
        print(f"Warning: No annotations found for subject {subject_id}. Skipping.")
        return None, None
    
    # We are interested in 'left_hand' and 'right_hand' events
    event_map = {
        'left_hand': event_id.get('T1'), 
        'right_hand': event_id.get('T2')
    }
    # Check if the standard BIDS events 'left_hand'/'right_hand' exist
    if event_map['left_hand'] is None or event_map['right_hand'] is None:
        bids_event_map = {'left_hand': event_id.get('left_hand'), 'right_hand': event_id.get('right_hand')}
        if bids_event_map['left_hand'] is not None and bids_event_map['right_hand'] is not None:
            event_map = bids_event_map
        else:
            print(f"Warning: Could not find T1/T2 or left/right hand events for subject {subject_id}. Found: {event_id.keys()}")
            return None, None

    # Pick only EEG channels
    raw.pick_types(eeg=True)

    # Apply bandpass filter
    raw.filter(l_freq=0.5, h_freq=50., verbose=False)

    # Create epochs
    tmin, tmax = -1., 4.  # extract 5-second epochs around the event
    epochs = mne.Epochs(raw, events, event_id=event_map, tmin=tmin, tmax=tmax, proj=False, 
                        baseline=None, preload=True, verbose=False)
    
    if len(epochs) == 0:
        print(f"Warning: No epochs found for the specified events for subject {subject_id}.")
        return None, None

    # Convert epochs to data array
    epoch_data = epochs.get_data() # shape: (n_epochs, n_channels, n_times)
    
    # Create spectrograms for each epoch
    fs = epochs.info['sfreq']
    nperseg = int(fs / 2) # 0.5-second window
    
    all_spectrograms = []
    for epoch in epoch_data:
        channel_spectrograms = []
        for channel_data in epoch:
            _, _, Zxx = stft(channel_data, fs=fs, nperseg=nperseg)
            channel_spectrograms.append(np.abs(Zxx))
        all_spectrograms.append(channel_spectrograms)
        
    labels = epochs.events[:, -1]
    # Map event IDs to 0 and 1
    labels = np.array([0 if label == event_map['left_hand'] else 1 for label in labels])

    return np.array(all_spectrograms), labels


if __name__ == '__main__':
    # Example usage:
    bids_root = 'eeg_stroke_patients'
    subject_id = '01'
    
    spectrograms, labels = get_eeg_spectrograms(bids_root, subject_id)
    
    if spectrograms is not None:
        print("Shape of spectrograms array:", spectrograms.shape)
        print("Shape of labels array:", labels.shape)
        print("Example labels:", labels[:10])
        print("Unique labels:", np.unique(labels))
    else:
        print(f"Failed to process data for subject {subject_id}")
