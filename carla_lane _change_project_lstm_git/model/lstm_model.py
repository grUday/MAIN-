# model/lstm_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LaneChangeLSTM(nn.Module):
    """
    LSTM model for lane change prediction.
    """
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.2, num_classes=3):
        """
        Initialize the LSTM model.
        
        Args:
            input_size (int): Number of input features
            hidden_size (int): Size of hidden LSTM layers
            num_layers (int): Number of LSTM layers
            dropout (float): Dropout rate
            num_classes (int): Number of output classes (left, none, right)
        """
        super(LaneChangeLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Linear(hidden_size, 1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size)
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)  # Shape: (batch_size, seq_len, hidden_size)
        
        # Apply attention to focus on important time steps
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)  # Shape: (batch_size, seq_len, 1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # Shape: (batch_size, hidden_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(context_vector))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def predict(self, x):
        """
        Make a prediction with the model.
        
        Args:
            x (torch.Tensor): Input tensor
            
        Returns:
            int: Predicted class (-1: left, 0: none, 1: right)
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(x)
            _, predicted = torch.max(outputs, 1)
            # Convert to -1, 0, 1 format
            predicted = predicted.item() - 1  # 0->-1, 1->0, 2->1
        return predicted