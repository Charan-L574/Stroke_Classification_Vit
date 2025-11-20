# EEG Sample CSV Files

This folder contains **real EEG sample patterns** extracted from the training dataset used to train the EEG classification model.

## Dataset Source

These samples are from the "5 Essential Words For Post-Stroke Patient EEG Dataset" - the actual data the model was trained on.

## Files

1. **class_0_sample.csv** - Pattern A: Low Activity
   - Mean amplitude: 4.46 μV
   - Standard deviation: 3.89
   - Range: 0-17 μV
   - Expected prediction: "Pattern A - Low Activity"
   - Characteristics: Low amplitude with minimal fluctuations

2. **class_1_sample.csv** - Pattern B: Moderate Activity
   - Mean amplitude: 6.81 μV
   - Standard deviation: 5.81
   - Range: 0-32 μV
   - Expected prediction: "Pattern B - Moderate Activity"
   - Characteristics: Moderate amplitude with balanced variability

3. **class_2_sample.csv** - Pattern C: High Amplitude
   - Mean amplitude: 14.23 μV
   - Standard deviation: 11.28
   - Range: 0-59 μV
   - Expected prediction: "Pattern C - High Amplitude"
   - Characteristics: High amplitude with significant variability

4. **class_3_sample.csv** - Pattern D: Very Low Activity
   - Mean amplitude: 2.11 μV
   - Standard deviation: 2.56
   - Range: 0-10 μV
   - Expected prediction: "Pattern D - Very Low Activity"
   - Characteristics: Very low amplitude, highly stable signals

5. **class_4_sample.csv** - Pattern E: Moderate-High Activity
   - Mean amplitude: 5.87 μV
   - Standard deviation: 6.39
   - Range: 0-35 μV
   - Expected prediction: "Pattern E - Moderate-High Activity"
   - Characteristics: Mixed amplitude with moderate variability

## How to Use

1. Go to the "Step 3: EEG Post-Stroke Monitoring" tab in the application
2. Either:
   - Click one of the pattern buttons (Pattern A, B, C, D, or E) to load sample data
   - Click "Upload EEG CSV File" and select one of these CSV files
3. Click "🧠 Analyze Brain Activity"
4. Review the predicted pattern classification

## CSV Format

Each CSV file contains:
- 256 rows of EEG signal values (downsampled from original 512 samples)
- Single column named 'eeg_value'
- Integer values representing EEG amplitude in microvolts (μV)

## Model Training

The EEG model was trained on 7,000 samples from this dataset:
- Class 0: 280 samples (4.0%)
- Class 1: 2,000 samples (28.6%)
- Class 2: 720 samples (10.3%)
- Class 3: 2,000 samples (28.6%)
- Class 4: 2,000 samples (28.6%)

Achieved **100% test accuracy** on the validation set.

## Note

These are real EEG signal samples from the training dataset. The model has been trained to recognize these specific patterns and classify new signals based on their similarity to these learned patterns.
