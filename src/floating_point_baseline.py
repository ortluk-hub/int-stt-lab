#!/usr/bin/env python3
"""
Floating-point baseline model for speech command recognition.
This serves as the reference implementation for Phase 2 comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FloatingPointLinear(nn.Module):
    """Standard floating-point linear layer."""
    def __init__(self, in_features, out_features):
        super(FloatingPointLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x):
        return self.linear(x)


class FloatingPointSpeechCommandNet(nn.Module):
    """A simple feed-forward network for speech command recognition with floating-point operations."""
    def __init__(self, input_size=40, hidden_size=64, num_classes=5):
        super(FloatingPointSpeechCommandNet, self).__init__()
        self.fc1 = FloatingPointLinear(input_size, hidden_size)
        self.fc2 = FloatingPointLinear(hidden_size, hidden_size)
        self.fc3 = FloatingPointLinear(hidden_size, num_classes)
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
    model = FloatingPointSpeechCommandNet(input_size=40, hidden_size=64, num_classes=5)
    print(f"Model: {model}")
    print(f"Trainable parameters: {count_parameters(model)}")
    
    # Example forward pass
    batch_size = 16
    x = torch.randn(batch_size, 40)  # random spectrogram patches
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("✅ Floating-point baseline STT model created successfully")