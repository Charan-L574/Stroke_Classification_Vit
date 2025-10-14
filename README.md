# Stroke Classification using Vision Transformer

This project implements a deep learning-based system to classify stroke types from brain MRI scans. It utilizes a fine-tuned Vision Transformer (ViT) model to achieve high-accuracy classification between Hemorrhagic, Ischemic, and Normal cases.

The final model achieves a validation accuracy of **99.19%**.

The project includes a user-friendly web interface built with Gradio for easy interaction and prediction.

## Features

- **High-Accuracy Model**: A Vision Transformer (`vit_base_patch16_224`) fine-tuned on a dataset of brain MRI scans.
- **Three-Class Classification**: Distinguishes between Hemorrhagic stroke, Ischemic stroke, and Normal brain scans.
- **Interactive Web UI**: A simple and intuitive interface powered by Gradio to upload an MRI scan and get an instant prediction.
- **GPU Accelerated**: The training and prediction processes are optimized to run on NVIDIA GPUs using PyTorch with CUDA.

## Model Performance

The model demonstrates excellent performance, achieving a final validation accuracy of 99.19%. The training and validation accuracy progressed as follows over 20 epochs:

![Accuracy Chart](accuracy_chart.png)

The training and validation loss also decreased consistently, indicating successful learning without significant overfitting.

![Loss Chart](loss_chart.png)

## Project Structure

```
.
├── MRI_DATA/               # Contains the dataset
├── src/
│   ├── prediction/
│   │   └── predict_image_only.py   # Prediction logic
│   └── training/
│       └── train_image_model_standalone.py # Model training script
├── app.py                  # The Gradio web application
├── requirements.txt        # Python dependencies
└── image_only_model_weights.pth # Trained model weights
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
python app.py
```

This will start a local web server. Open the provided URL in your browser to access the application. You can then upload an MRI image to get a classification result.

## Model Training

The model was trained using the script `src/training/train_image_model_standalone.py`. This script handles data loading, augmentation, model fine-tuning, and saving the final weights.

To retrain the model, you can run:
```bash
python src/training/train_image_model_standalone.py
```
Make sure the `MRI_DATA` directory is structured correctly before starting the training.

## Technologies Used

- **PyTorch**: The core deep learning framework.
- **Timm (PyTorch Image Models)**: For instantiating the pre-trained Vision Transformer model.
- **Gradio**: For building the interactive web demo.
- **Scikit-learn**: For data splitting.
- **Pillow**: For image processing.
