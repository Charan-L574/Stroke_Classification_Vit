import torch
import torch.nn as nn

from ..models.image_model import create_image_model
from ..models.clinical_model import ClinicalModel

class FusionModel(nn.Module):
    def __init__(self, image_model_weights, clinical_model_weights, clinical_input_size, num_classes=4, dropout_rate=0.3):
        """
        Bimodal fusion network for image and clinical data.

        Args:
            image_model_weights (str): Path to the saved weights of the image model.
            clinical_model_weights (str): Path to the saved weights of the clinical model.
            clinical_input_size (int): The number of input features for the clinical model.
            num_classes (int, optional): Number of output classes. Defaults to 4.
            dropout_rate (float, optional): Dropout rate for the classification head. Defaults to 0.3.
        """
        super(FusionModel, self).__init__()
        
        # Load pre-trained models
        self.image_model = create_image_model(num_classes=num_classes)
        # The image model from timm doesn't have a separate feature extractor, so we'll take the output before the head.
        image_model_state_dict = torch.load(image_model_weights)
        # Adjust for the number of classes the pretrained model was saved with
        num_pretrained_classes = image_model_state_dict['head.weight'].shape[0]
        self.image_model = create_image_model(num_classes=num_pretrained_classes)
        self.image_model.load_state_dict(image_model_state_dict)
        vit_feature_size = self.image_model.head.in_features
        self.image_model.head = nn.Identity() # Remove classification head
        
        self.clinical_model = ClinicalModel(input_size=clinical_input_size, hidden_sizes=[64, 32, 16], feature_vector_size=16)
        self.clinical_model.load_state_dict(torch.load(clinical_model_weights, weights_only=True))
        
        # Fine-tuning: Unfreeze the backbones to allow them to be trained
        for param in self.image_model.parameters():
            param.requires_grad = True
        for param in self.clinical_model.parameters():
            param.requires_grad = True
            
        # Determine the size of the concatenated feature vector
        clinical_feature_size = 16
        concatenated_size = vit_feature_size + clinical_feature_size
        
        # Create the classification head
        self.fusion_head = nn.Sequential(
            nn.Linear(concatenated_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, image_data, clinical_data):
        """
        Forward pass of the fusion model.

        Args:
            image_data (torch.Tensor): Input image tensor.
            clinical_data (torch.Tensor): Input clinical data tensor.

        Returns:
            torch.Tensor: The output logits.
        """
        # Get feature embeddings from each branch
        image_features = self.image_model(image_data)
        clinical_features = self.clinical_model(clinical_data)
        
        # Concatenate the features
        concatenated_features = torch.cat((image_features, clinical_features), dim=1)
        
        # Pass through the fusion head
        output = self.fusion_head(concatenated_features)
        
        return output

if __name__ == '__main__':
    # This is a placeholder for where the actual weights would be.
    # We need to run the individual training scripts first to generate these.
    # For now, let's save dummy weights for testing the architecture.
    
    import os
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save dummy weights for each model
    torch.save(create_image_model(num_classes=3).state_dict(), 'models/image_model_weights.pth')
    torch.save(ClinicalModel(input_size=25, hidden_sizes=[64, 32, 16], feature_vector_size=16).state_dict(), 'models/clinical_model_weights.pth')

    # Initialize the fusion model
    fusion_model = FusionModel(
        image_model_weights='models/image_model_weights.pth',
        clinical_model_weights='models/clinical_model_weights.pth',
        clinical_input_size=25,
        num_classes=3 # Matching the image data classes
    )
    
    print(fusion_model)
    
    # Test with dummy inputs
    batch_size = 4
    image_input = torch.randn(batch_size, 3, 224, 224)
    clinical_input = torch.randn(batch_size, 25)
    
    output = fusion_model(image_input, clinical_input)
    print("Output shape:", output.shape)
    assert output.shape == (batch_size, 3)
