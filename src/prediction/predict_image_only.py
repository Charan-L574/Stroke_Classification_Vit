import torch
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import timm
import numpy as np
import json
import sys

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data_preprocessing.image_preprocessing import get_image_transforms

def predict_image_only(image_path):
    """
    Predicts the stroke type from a single image using the trained image-only model.

    Args:
        image_path (str or Path): The path to the input image.

    Returns:
        dict: A dictionary containing the predicted class and the class probabilities.
    """
    print("--- Starting Image-Only Prediction ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Configuration ---
    MODEL_PATH = project_root / 'src' / 'prediction' / 'image_only_model_weights.pth'
    IMAGE_SIZE = 224
    class_names = sorted(['Haemorrhagic', 'Ischemic', 'Normal'])
    num_classes = len(class_names)

    if not Path(MODEL_PATH).exists():
        raise FileNotFoundError(f"Model weights not found at {MODEL_PATH}")

    # --- Model Initialization ---
    print("Loading image-only model...")
    model = timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=num_classes)
    
    try:
        # Load the state dict, ignoring potential size mismatches if any
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=True)
    except RuntimeError as e:
        print(f"Caught a RuntimeError, which can happen with timm models. Attempting to load again.")
        # This can sometimes happen if the model was saved in a slightly different version.
        # Loading with strict=False is a common workaround, but we'll try strict=True first.
        # For this case, we will trust the saved model structure.
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

    model.to(device)
    model.eval()
    print("Model loaded successfully.")

    # --- Image Preprocessing ---
    print("Loading and preprocessing image...")
    try:
        image = Image.open(image_path).convert('RGB')
        transform = get_image_transforms(IMAGE_SIZE)['val']
        image_tensor = transform(image).unsqueeze(0).to(device)
        print("Image data processed correctly.")
    except Exception as e:
        return {"error": f"Failed to process image: {e}"}

    # --- Prediction ---
    print("Making prediction...")
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)
        _, predicted_idx = torch.max(outputs, 1)

    predicted_class = class_names[predicted_idx.item()]
    
    # Create a dictionary of class probabilities
    prob_dict = {class_names[i]: probabilities[0][i].item() for i in range(num_classes)}
    
    print("--- Prediction Complete ---")
    
    result = {
        "predicted_class": predicted_class,
        "probabilities": prob_dict
    }
    return result

def main():
    """
    Main function to run a test prediction.
    """
    # Use a test image from the dataset for validation
    # This path assumes the script is run from the root of the project
    test_image_path = project_root / 'MRI_DATA' / 'Stroke_classification' / 'Ischemic' / 'Ellappan T2-12.jpg_Ischemic_10.png'
    
    if not test_image_path.exists():
        print(f"Test image not found at {test_image_path}. Please check the path.")
        return

    prediction = predict_image_only(test_image_path)

    print("\n--- FINAL RESULT ---")
    if "error" in prediction:
        print(f"An error occurred: {prediction['error']}")
    else:
        predicted_class = prediction['predicted_class']
        probabilities = prediction['probabilities']
        
        print(f"Predicted Class: {predicted_class}")
        print(f"Probabilities: {json.dumps(probabilities, indent=4)}")

        # Simple validation based on folder name
        actual_class = test_image_path.parent.name
        if predicted_class == actual_class:
            print("\nValidation: ✅ Example prediction matches the image's source folder.")
        else:
            print(f"\nValidation: ❌ Example prediction does NOT match. Actual was '{actual_class}'.")


if __name__ == '__main__':
    main()
