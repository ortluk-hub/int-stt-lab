import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSpeechCommandNet(nn.Module):
    """
    A simple feed-forward network for speech command recognition.
    Takes spectrogram patches as input and outputs class probabilities.
    """
    def __init__(self, input_size=40, hidden_size=128, num_classes=10):
        super(SimpleSpeechCommandNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x: batch of spectrogram patches (batch_size, input_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Example usage
    model = SimpleSpeechCommandNet(input_size=40, hidden_size=64, num_classes=5)
    print(f"Model: {model}")
    print(f"Trainable parameters: {count_parameters(model)}")
    
    # Example forward pass
    batch_size = 16
    x = torch.randn(batch_size, 40)  # random spectrogram patches
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Simple STT model created successfully")
