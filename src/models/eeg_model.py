import torch
import torch.nn as nn

class EEGModel(nn.Module):
    def __init__(self, num_channels, num_freq_bins, cnn_out_channels=64, cnn_kernel_size=5, pool_size=2, lstm_hidden_size=128, num_lstm_layers=2, feature_vector_size=64):
        """
        EEG model with 1D CNN and LSTM layers.

        Args:
            num_channels (int): Number of EEG channels.
            num_freq_bins (int): Number of frequency bins in the spectrogram.
            cnn_out_channels (int): Number of output channels for the CNN layer.
            cnn_kernel_size (int): Kernel size for the CNN layer.
            pool_size (int): Kernel size for the max pooling layer.
            lstm_hidden_size (int): Number of features in the hidden state of the LSTM.
            num_lstm_layers (int): Number of recurrent layers in the LSTM.
            feature_vector_size (int): The size of the final output feature vector.
        """
        super(EEGModel, self).__init__()
        
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=num_freq_bins, out_channels=cnn_out_channels, kernel_size=cnn_kernel_size, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=pool_size)
        )
        
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels, 
            hidden_size=lstm_hidden_size, 
            num_layers=num_lstm_layers, 
            batch_first=True,
            bidirectional=True # Using a bidirectional LSTM
        )
        
        self.fc = nn.Linear(lstm_hidden_size * 2 * num_channels, feature_vector_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, num_freq_bins, time_steps)

        Returns:
            torch.Tensor: Output feature vector of shape (batch_size, feature_vector_size)
        """
        batch_size, num_channels, num_freq_bins, time_steps = x.shape
        
        x = x.reshape(batch_size * num_channels, num_freq_bins, time_steps)
        
        cnn_out = self.cnn(x)
        
        lstm_input = cnn_out.permute(0, 2, 1)
        
        _, (hidden, _) = self.lstm(lstm_input)
        
        forward_hidden = hidden[-2, :, :]
        backward_hidden = hidden[-1, :, :]
        
        lstm_out = torch.cat((forward_hidden, backward_hidden), dim=1)
        
        x = lstm_out.reshape(batch_size, -1)
        
        feature_vector = self.fc(x)
        feature_vector = self.relu(feature_vector)
        
        return feature_vector

if __name__ == '__main__':
    # Example usage:
    batch_size = 4
    num_channels = 2
    num_freq_bins = 65 
    time_steps = 44 
    
    eeg_model = EEGModel(num_channels=num_channels, num_freq_bins=num_freq_bins, feature_vector_size=64)
    print(eeg_model)
    
    dummy_input = torch.randn(batch_size, num_channels, num_freq_bins, time_steps)
    output = eeg_model(dummy_input)
    print("Output shape:", output.shape)
    assert output.shape == (batch_size, 64)