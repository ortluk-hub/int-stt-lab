#!/usr/bin/env python3
"""
Integer-only STT model prototype.
This version demonstrates integer-only operations by quantizing weights and activations to int8.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

def quantize_tensor(tensor, scale=127.0):
    """Quantize a tensor to int8 range [-128, 127] and dequantize back to float for simulation."""
    # In a real integer-only framework, we would keep the tensor as int8
    # For simulation, we quantize and dequantize to show the effect
    quantized = (tensor * scale).round().clamp(-128, 127)
    dequantized = quantized / scale
    return dequantized

class IntegerOnlyLinear(nn.Module):
    """Linear layer with integer-only weights and activations (simulated)."""
    def __init__(self, in_features, out_features):
        super(IntegerOnlyLinear, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        # In a real integer-only framework, weights would be stored as int8
        # We'll simulate this by quantizing the weights during forward pass
        
    def forward(self, x):
        # Quantize weights to int8 (simulated)
        quantized_weight = quantize_tensor(self.linear.weight)
        quantized_bias = quantize_tensor(self.linear.bias) if self.linear.bias is not None else None
        
        # Perform linear operation with quantized weights
        # In a real integer-only framework, this would be integer arithmetic
        x = F.linear(x, quantized_weight, quantized_bias)
        
        # Quantize activation to int8 (simulated)
        x = quantize_tensor(x)
        return x

class SimpleSpeechCommandNet(nn.Module):
    """A simple feed-forward network for speech command recognition with integer-only operations."""
    def __init__(self, input_size=40, hidden_size=128, num_classes=10):
        super(SimpleSpeechCommandNet, self).__init__()
        self.fc1 = IntegerOnlyLinear(input_size, hidden_size)
        self.fc2 = IntegerOnlyLinear(hidden_size, hidden_size)
        self.fc3 = IntegerOnlyLinear(hidden_size, num_classes)
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
    print("✅ Integer-only STT model created successfully")
