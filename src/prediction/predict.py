import torch
import pandas as pd
import joblib
import os
from PIL import Image

from ..data_preprocessing.image_preprocessing import get_image_transforms
from ..models.fusion_model import FusionModel


def predict_stroke(image_path, clinical_data_row, model_dir='models/'):
    """
    Predicts stroke type using the trained bimodal fusion model.

    Args:
        image_path (str): Path to the MRI image file.
        clinical_data_row (dict): A dictionary representing a single row of clinical data.
                                  Must contain all features the model was trained on.
        model_dir (str, optional): Directory containing model and preprocessor files.
                                   Defaults to 'models/'.

    Returns:
        str: The predicted class name (e.g., 'Normal', 'Ischemic', 'Haemorrhagic').
        dict: A dictionary containing the raw prediction probabilities.
    """
    print("--- Starting Prediction ---")

    model_path = os.path.join(model_dir, 'fusion_model_weights.pth')
    preprocessor_path = os.path.join(model_dir, 'clinical_data_preprocessor.joblib')
    image_model_weights_path = os.path.join(model_dir, 'image_model_weights.pth')
    clinical_model_weights_path = os.path.join(model_dir, 'clinical_model_weights.pth')

    # --- Validation ---
    if not all(os.path.exists(p) for p in [image_path, model_path, preprocessor_path, image_model_weights_path, clinical_model_weights_path]):
        missing = [p for p in [image_path, model_path, preprocessor_path, image_model_weights_path, clinical_model_weights_path] if not os.path.exists(p)]
        raise FileNotFoundError(f"One or more required files are missing: {', '.join(missing)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Clinical Preprocessor and Process Data ---
    print("Loading clinical data preprocessor and processing data...")
    try:
        preprocessor = joblib.load(preprocessor_path)
    except Exception as e:
        raise RuntimeError(f"Error loading preprocessor: {e}")

    clinical_df = pd.DataFrame([clinical_data_row])

    try:
        processed_clinical_data = preprocessor.transform(clinical_df)
        clinical_tensor = torch.tensor(processed_clinical_data, dtype=torch.float32).to(device)
        clinical_input_size = processed_clinical_data.shape[1]
        print(f"Clinical data processed. Feature size: {clinical_input_size}")
    except Exception as e:
        raise ValueError(f"Error processing clinical data: {e}. Ensure input dictionary matches training columns.")

    # --- Load Image and Preprocess ---
    print("Loading and preprocessing image...")
    try:
        image = Image.open(image_path).convert('RGB')
        # CRITICAL FIX: Use the 'val' transforms from the training pipeline
        # to ensure normalization is consistent with what the model was trained on.
        transforms = get_image_transforms()['val']
        image_tensor = transforms(image).unsqueeze(0).to(device)  # Add batch dimension
        print("Image data processed correctly.")
    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")

    # --- Load Model ---
    print("Loading fusion model...")
    num_classes = 3  # 'Haemorrhagic', 'Ischemic', 'Normal'

    model = FusionModel(
        image_model_weights=image_model_weights_path,
        clinical_model_weights=clinical_model_weights_path,
        clinical_input_size=clinical_input_size,
        num_classes=num_classes
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- Prediction ---
    print("Making prediction...")
    with torch.no_grad():
        output = model(image_tensor, clinical_tensor)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        _, predicted_idx = torch.max(output, 1)

    # --- Map Prediction to Class Name ---
    # The standard order is alphabetical, which is how ImageFolder reads them.
    class_names = sorted(['Haemorrhagic', 'Ischemic', 'Normal']) # Ensures correct order
    predicted_class = class_names[predicted_idx.item()]
    
    probs_dict = {class_names[i]: probabilities[0][i].item() for i in range(len(class_names))}

    print("--- Prediction Complete ---")
    return predicted_class, probs_dict

if __name__ == '__main__':
    # --- Example Usage ---
    # This example demonstrates how to use the predict_stroke function.
    # You would replace this with your actual new data.

    # 1. Define the path to a new image you want to classify.
    example_image_path = 'MRI_DATA/Stroke_classification/Ischemic/Ellappan DWI-16.jpg_Ischemic_1.png'

    # 2. Define the clinical data for the corresponding patient.
    # This must be a dictionary with the same feature names as in the original CSV.
    example_clinical_data = {
        'gender': 'Female',
        'age': 61.0,
        'hypertension': 0,
        'heart_disease': 0,
        'ever_married': 'Yes',
        'work_type': 'Self-employed',
        'Residence_type': 'Rural',
        'avg_glucose_level': 202.21,
        'bmi': 28.89, # Using a plausible value, as original might be missing
        'smoking_status': 'never smoked'
    }

    try:
        # 3. Call the prediction function
        prediction, probabilities = predict_stroke(example_image_path, example_clinical_data)
        print(f"\n--- FINAL RESULT ---")
        print(f"Predicted Class: {prediction}")
        print(f"Probabilities: {probabilities}")

        # A simple check for the example
        if 'Ischemic' in example_image_path and prediction == 'Ischemic':
            print("\nValidation: Example prediction matches the image's source folder. Correct!")
        else:
            print("\nValidation: Example prediction does not match the image's source folder.")

    except (FileNotFoundError, ValueError, RuntimeError) as e:
        print(f"\n--- PREDICTION FAILED ---")
        print(f"An error occurred: {e}")
        print("Please ensure all model files exist in the 'models/' directory and the input data is correct.")
