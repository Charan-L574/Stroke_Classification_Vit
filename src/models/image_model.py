import torch
import torch.nn as nn
import timm

def create_image_model(num_classes=4):
    """
    Creates an image classification model using a pre-trained Vision Transformer (ViT).

    Args:
        num_classes (int, optional): The number of output classes. Defaults to 4.

    Returns:
        torch.nn.Module: The image model.
    """
    # Load a pre-trained ViT model
    model = timm.create_model('vit_base_patch16_224', pretrained=True)
    
    # Replace the classification head with a new one for our number of classes
    n_features = model.head.in_features
    model.head = nn.Linear(n_features, num_classes)
    
    return model

if __name__ == '__main__':
    # Example usage:
    image_model = create_image_model()
    print(image_model)
    
    # Test with a dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = image_model(dummy_input)
    print("Output shape:", output.shape)
