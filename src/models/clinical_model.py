import torch
import torch.nn as nn

class ClinicalModel(nn.Module):
    def __init__(self, input_size, hidden_sizes, feature_vector_size):
        """
        MLP model for clinical data.

        Args:
            input_size (int): The number of input features.
            hidden_sizes (list): A list of integers specifying the size of each hidden layer.
            feature_vector_size (int): The size of the final output feature vector.
        """
        super(ClinicalModel, self).__init__()
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            prev_size = hidden_size
            
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(prev_size, feature_vector_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output feature vector of shape (batch_size, feature_vector_size).
        """
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        x = self.relu(x)
        return x

if __name__ == '__main__':
    # Example usage:
    # The input size depends on the preprocessed clinical data
    # Let's assume it's 25 after one-hot encoding and other steps
    input_size = 25 
    hidden_sizes = [64, 32, 16]
    feature_vector_size = 16 # As per prompt
    
    clinical_model = ClinicalModel(input_size, hidden_sizes, feature_vector_size)
    print(clinical_model)
    
    # Test with a dummy input
    batch_size = 4
    dummy_input = torch.randn(batch_size, input_size)
    output = clinical_model(dummy_input)
    print("Output shape:", output.shape)
    assert output.shape == (batch_size, feature_vector_size)
