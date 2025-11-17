import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM) for model interpretability.
    This implementation works with Vision Transformer (ViT) models.
    """
    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM.
        
        Args:
            model: The PyTorch model
            target_layer: The layer to compute gradients for (usually the last convolutional layer)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        """Hook to save the forward pass activations"""
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save the gradients during backward pass"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_tensor, class_idx=None):
        """
        Generate the Grad-CAM heatmap.
        
        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index. If None, uses the predicted class.
            
        Returns:
            cam: The Grad-CAM heatmap
            predicted_class: The predicted class index
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(input_tensor)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for the target class
        one_hot = torch.zeros_like(output)
        one_hot[0][class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (num_patches, feature_dim) for ViT
        activations = self.activations[0]  # (num_patches, feature_dim) for ViT
        
        # For ViT, we need to handle the CLS token
        # Remove the CLS token (first token)
        if len(activations.shape) == 2:
            activations = activations[1:]  # Remove CLS token
            gradients = gradients[1:]
        
        # Calculate weights by global average pooling of gradients
        weights = gradients.mean(dim=0)  # Shape: (feature_dim,) e.g. (768,)
        
        # Weighted combination of activation maps
        # activations shape: (num_patches, feature_dim) e.g. (196, 768)
        # weights shape: (feature_dim,) e.g. (768,)
        # We want to weight each patch's features and sum across features
        cam = torch.matmul(activations, weights)  # Shape: (num_patches,) e.g. (196,)
        
        # Apply ReLU to focus on positive influence
        cam = F.relu(cam)
        
        # Normalize
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy(), class_idx
    
    def visualize(self, input_image, input_tensor, class_idx=None, alpha=0.4):
        """
        Create a visualization overlaying the Grad-CAM on the original image.
        
        Args:
            input_image: Original PIL image
            input_tensor: Preprocessed input tensor
            class_idx: Target class. If None, uses predicted class.
            alpha: Transparency of the heatmap overlay
            
        Returns:
            overlayed_image: PIL Image with Grad-CAM overlay
            predicted_class: The predicted class index
        """
        # Generate CAM
        cam, predicted_class = self.generate_cam(input_tensor, class_idx)
        
        # Resize CAM to match input image size
        # For ViT, we need to reshape cam to 2D first
        patch_size = 16  # Standard for ViT-base
        num_patches = int(np.sqrt(len(cam)))
        cam_2d = cam.reshape(num_patches, num_patches)
        
        # Apply threshold to focus on important regions (remove weak activations)
        threshold = 0.3  # Only show regions with activation > 30%
        cam_2d_thresholded = np.where(cam_2d > threshold, cam_2d, 0)
        
        # Renormalize after thresholding
        if cam_2d_thresholded.max() > 0:
            cam_2d_thresholded = (cam_2d_thresholded - cam_2d_thresholded.min()) / (cam_2d_thresholded.max() - cam_2d_thresholded.min())
        
        # Resize to input image size
        input_size = input_image.size
        cam_resized = cv2.resize(cam_2d_thresholded, input_size)
        
        # Create heatmap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert original image to numpy
        original_image = np.array(input_image)
        
        # Overlay
        overlayed = heatmap * alpha + original_image * (1 - alpha)
        overlayed = np.uint8(overlayed)
        
        return Image.fromarray(overlayed), predicted_class


def explain_image_prediction(model, image_path, device='cpu'):
    """
    Generate Grad-CAM explanation for an image prediction.
    
    Args:
        model: Trained Vision Transformer model
        image_path: Path to the input image
        device: Device to run on ('cpu' or 'cuda')
        
    Returns:
        tuple: (original_image, explained_image, predicted_class, class_confidence)
    """
    from src.data_preprocessing.image_preprocessing import get_image_transforms
    
    # Load and preprocess image
    transform = get_image_transforms()['val']
    original_image = Image.open(image_path).convert('RGB')
    input_tensor = transform(original_image).unsqueeze(0).to(device)
    
    # For ViT models, we target the last layer before the head
    # This is typically model.blocks[-1].norm1 or a similar layer
    target_layer = model.blocks[-1].norm1
    
    # Create Grad-CAM instance
    grad_cam = GradCAM(model, target_layer)
    
    # Generate visualization
    explained_image, predicted_class = grad_cam.visualize(original_image, input_tensor)
    
    # Get confidence
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = F.softmax(output, dim=1)
        class_confidence = probabilities[0][predicted_class].item()
    
    return original_image, explained_image, predicted_class, class_confidence


if __name__ == '__main__':
    print("Grad-CAM utilities for explainable AI ready.")
    print("Use the 'explain_image_prediction' function to generate explanations for image predictions.")
