# Multi-Modal Stroke Classification System

This project implements an advanced deep learning-based system for comprehensive stroke risk assessment and diagnosis using multiple modalities:
- **Brain MRI/CT Scan Analysis** - Vision Transformer for stroke detection
- **Clinical Biomarker Analysis** - Neural network for risk prediction from blood markers
- **EEG Signal Analysis** - 1D CNN for brain state classification
- **Multi-Modal Fusion** - Combined analysis from all three modalities

The system achieves excellent performance with MRI classification at **99.19%** accuracy, biomarker prediction at **84.17%** recall, and EEG classification at **100%** accuracy.

The project includes an interactive web interface built with Gradio featuring explainable AI through Grad-CAM visualizations.

## Features

### Multi-Modal Analysis
- **Brain MRI/CT Scan Classification**: Vision Transformer (`vit_base_patch16_224`) for three-class classification (Hemorrhagic, Ischemic, Normal)
- **Clinical Biomarker Risk Assessment**: MLP neural network analyzing 29 clinical features from 114K patient records
- **EEG Signal Analysis**: 1D CNN for 5-class brain state classification (Normal, Drowsy, Sleep, Seizure, Epilepsy)
- **Fusion Analysis**: Combined predictions from all three modalities for comprehensive assessment

### Explainable AI
- **Grad-CAM Visualization**: Visual heatmaps showing which brain regions influenced the MRI diagnosis
- **Risk Factor Analysis**: Detailed breakdown of biomarker contributions to stroke risk
- **Personalized Prevention Routines**: AI-generated recommendations based on individual risk factors

### Advanced Dataset Processing
- **Biomarker Models**: Trained on 114,081 samples from 3 merged datasets with 29 features
- **EEG Classification**: 7,000 samples with 5 brain state classes extracted from post-stroke patient data
- **Class Imbalance Handling**: Weighted loss functions (pos_weight=21.03) for accurate minority class prediction

### User Experience
- **Interactive Web UI**: Clean, professional Gradio interface with side-by-side result displays
- **Real-time Predictions**: Instant classification and risk assessment
- **Example Patterns**: Pre-loaded EEG patterns (Normal, Drowsy, Sleep) for testing
- **GPU Accelerated**: Optimized for NVIDIA GPUs using PyTorch with CUDA

## Model Performance

### Brain MRI/CT Classification
- **Validation Accuracy**: 99.19%
- **Architecture**: Vision Transformer (vit_base_patch16_224)
- **Classes**: Hemorrhagic, Ischemic, Normal

### Clinical Biomarker Risk Prediction
- **Recall**: 84.17% (optimized for detecting high-risk patients)
- **Training Data**: 114,081 samples from 3 merged datasets
- **Features**: 29 clinical and lifestyle factors
- **Class Imbalance**: 21:1 ratio handled with weighted BCE loss

### EEG Signal Classification
- **Validation Accuracy**: 100%
- **Architecture**: 1D CNN with 265,701 parameters
- **Classes**: Normal, Drowsy, Sleep, Seizure Activity, Epilepsy
- **Training Data**: 7,000 EEG samples from post-stroke patients

## Project Structure

```
.
├── app_advanced.py                 # Main Gradio web application
├── requirements.txt                # Python dependencies
├── dataset/                        # Training datasets
│   ├── healthcare-dataset-stroke-data.csv
│   ├── dataset.csv
│   ├── diabetes_data.csv
│   └── 5 Essential Words For Post-Stroke Patient EEG Dataset.csv
├── src/
│   ├── data_preprocessing/
│   │   ├── biomarker_data_preprocessing_enhanced.py  # Merges 3 biomarker datasets
│   │   ├── eeg_data_preprocessing.py                 # EEG signal processing
│   │   └── image_preprocessing.py                    # MRI/CT preprocessing
│   ├── models/
│   │   ├── biomarker_model.py      # MLP for clinical risk
│   │   ├── eeg_model.py            # 1D CNN for EEG
│   │   ├── image_model.py          # Vision Transformer
│   │   └── fusion_model.py         # Multi-modal fusion
│   ├── training/
│   │   ├── train_biomarker_model_enhanced.py  # Train on 114K samples
│   │   ├── train_eeg_model_enhanced.py        # Train on 7K EEG samples
│   │   └── train_image_model_standalone.py    # Train ViT model
│   ├── prediction/
│   │   ├── biomarker_model_weights.pth        # Trained biomarker model
│   │   ├── biomarker_model_weights_preprocessor.pkl  # Feature preprocessor
│   │   ├── eeg_model_weights.pth              # Trained EEG model
│   │   └── image_only_model_weights.pth       # Trained MRI model
│   └── utils/
│       ├── gradcam.py              # Grad-CAM explainability
│       ├── plot_charts.py          # Visualization utilities
│       └── shap_explainer.py       # SHAP analysis
├── assets/                         # UI images and resources
├── unused_files/                   # Archived development files
└── *.md                            # Documentation files
```

## Getting Started

### Prerequisites

- Python 3.8+
- NVIDIA GPU with CUDA support (recommended)
- Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Charan-L574/Stroke_Classification_Vit.git
    cd Stroke_Classification_Vit
    ```

2.  **Install the required Python packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

To launch the Gradio web interface, run the following command in your terminal:

```bash
python app_advanced.py
```

This will start a local web server (typically at `http://localhost:7860`). Open the provided URL in your browser to access the application.

The interface includes four main tabs:
1. **ℹ️ Information** - Overview and instructions
2. **🩺 Clinical Biomarker Analysis** - Enter patient data for stroke risk assessment
3. **🧠 Brain MRI/CT Scan Analysis** - Upload brain scan for classification with Grad-CAM visualization
4. **📊 EEG Signal Analysis** - Input EEG signals for brain state classification
5. **🔬 Multi-Modal Fusion Analysis** - Combined analysis from all three modalities

## Model Training

### Training the Biomarker Model
```bash
python src/training/train_biomarker_model_enhanced.py
```
- Merges 3 datasets (114,081 samples)
- Trains MLP with BCEWithLogitsLoss (pos_weight=21.03)
- 100 epochs with ReduceLROnPlateau scheduler
- Saves model and preprocessor separately

### Training the EEG Model
```bash
python src/training/train_eeg_model_enhanced.py
```
- Processes 7,000 EEG samples with 5 brain state classes
- 1D CNN architecture with 265,701 parameters
- 50 epochs with class-weighted CrossEntropyLoss
- Achieves 100% validation accuracy

### Training the Image Model
```bash
python src/training/train_image_model_standalone.py
```
- Fine-tunes Vision Transformer on MRI/CT scans
- Three-class classification (Hemorrhagic, Ischemic, Normal)
- Data augmentation and GPU acceleration
- Achieves 99.19% validation accuracy

**Note**: Ensure all datasets are in the `dataset/` directory before training.

## Technologies Used

- **PyTorch 2.8.0**: Core deep learning framework with CUDA support
- **Timm (PyTorch Image Models)**: Pre-trained Vision Transformer models
- **Gradio 4.x**: Interactive web interface with Row/Column layouts
- **Scikit-learn 1.7.2**: Preprocessing, feature scaling, and data splitting
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Matplotlib**: Grad-CAM visualization and plotting
- **Pillow**: Image processing and augmentation

## Documentation

- **TRAINING_SUMMARY.md** - Detailed training process and results
- **BIOMARKER_3_DATASETS_EXPLANATION.md** - Dataset merging methodology
- **EEG_5_CLASSES_EXPLANATION.md** - EEG label extraction process
- **EEG_MODEL_EXPLANATION.md** - Model architecture details
- **ADVANCED_FEATURES.md** - Feature documentation
- **CLEANUP_RECOMMENDATIONS.md** - Workspace maintenance guide

## Key Features Explained

### Grad-CAM Visualization
The system uses Gradient-weighted Class Activation Mapping (Grad-CAM) to highlight which regions of the brain scan influenced the model's decision. This provides transparency and helps medical professionals understand the AI's reasoning.

### Clinical Biomarker Analysis
Analyzes 29 clinical features including:
- Demographics (age, gender)
- Vital signs (blood pressure, BMI)
- Medical history (heart disease, diabetes, high cholesterol)
- Lifestyle factors (smoking, physical activity)
- Lab values (glucose levels, biomarkers)

### Multi-Modal Fusion
Combines predictions from all three modalities with configurable weights:
- **Image Model**: Primary predictor for stroke detection
- **Biomarker Model**: Risk assessment from clinical data
- **EEG Model**: Brain state classification

The fusion model provides a comprehensive view of the patient's condition by integrating multiple data sources.

## Performance Metrics

| Model | Metric | Value |
|-------|--------|-------|
| MRI/CT Classification | Accuracy | 99.19% |
| Biomarker Risk | Recall | 84.17% |
| Biomarker Risk | Accuracy | 66.95% |
| EEG Classification | Accuracy | 100% |
| EEG Classification | All Classes | 100% |

## License

This project is for educational and research purposes.
