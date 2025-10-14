import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def get_image_transforms(image_size=224):
    """
    Returns a dictionary of transforms for training and validation.
    Includes more aggressive data augmentation for the training set.
    """
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

def preprocess_image(image_path):
    """
    Loads an image, resizes it, and applies validation transformations.

    Args:
        image_path (str): The path to the image file.

    Returns:
        torch.Tensor: The preprocessed image tensor.
    """
    try:
        pil_image = Image.open(image_path).convert('RGB')
    except FileNotFoundError:
        raise FileNotFoundError(f"Image not found at {image_path}")
    
    transform = get_image_transforms()['val']
    
    return transform(pil_image)

if __name__ == '__main__':
    # Example usage:
    # Create a dummy image for testing
    dummy_image_array = np.random.randint(0, 256, size=(100, 100, 3), dtype=np.uint8)
    dummy_image = Image.fromarray(dummy_image_array)
    dummy_image.save("dummy_image.png")
    
    try:
        preprocessed_tensor = preprocess_image("dummy_image.png")
        print("Shape of preprocessed tensor:", preprocessed_tensor.shape)
        assert preprocessed_tensor.shape == (3, 224, 224)
        print("Preprocessing test passed.")
    except FileNotFoundError as e:
        print(e)
    finally:
        import os
        os.remove("dummy_image.png")
