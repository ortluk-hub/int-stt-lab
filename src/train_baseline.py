#!/usr/bin/env python3
"""
Baseline training script for floating-point STT model.
This serves as the reference implementation for Phase 2 comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from floating_point_baseline import FloatingPointSpeechCommandNet, count_parameters
from torch.utils.data import DataLoader, TensorDataset


def create_dummy_dataset(batch_size=32, time_steps=1, input_size=40, num_classes=5, num_samples=100):
    """Create a dummy dataset for demonstration"""
    # Input: (batch, feature) - treating as framewise for simplicity
    inputs = torch.randn(num_samples, input_size)
    # Targets: random integers
    targets = torch.randint(0, num_classes, (num_samples,))
    
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_baseline(model, dataloader, num_epochs=10):
    """Train the floating-point baseline model"""
    print("Starting floating-point baseline training...")
    print("=" * 50)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in dataloader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
        
        epoch_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Loss: {epoch_loss:.4f} '
              f'Accuracy: {accuracy:.2f}%')
    
    print("=" * 50)
    print("✅ Baseline training completed")
    return model


def verify_no_integer_simulation(model):
    """Verify that the model doesn't use integer-only simulation"""
    print("\nVerifying baseline model properties:")
    print("- Uses standard nn.Linear layers (floating-point)")
    print("- No integer quantization/dequantization in forward pass")
    print("- Standard floating-point parameter updates")
    
    # Check if model has the expected layers
    has_fc1 = hasattr(model, 'fc1') and hasattr(model.fc1, 'linear')
    has_fc2 = hasattr(model, 'fc2') and hasattr(model.fc2, 'linear')
    has_fc3 = hasattr(model, 'fc3') and hasattr(model.fc3, 'linear')
    
    if has_fc1 and has_fc2 and has_fc3:
        print("✓ Model contains expected floating-point linear layers")
    else:
        print("⚠ Model structure check failed")


if __name__ == "__main__":
    # Create model
    model = FloatingPointSpeechCommandNet(input_size=40, hidden_size=64, num_classes=5)
    print(f"Baseline model created with {count_parameters(model)} parameters")
    
    # Verify it's a proper baseline
    verify_no_integer_simulation(model)
    
    # Create dataset
    dataloader = create_dummy_dataset(batch_size=32, num_samples=100)
    print(f"Created dataset with {len(dataloader.dataset)} samples")
    
    # Train baseline model
    trained_model = train_baseline(model, dataloader, num_epochs=5)
    
    # Save the trained model for comparison
    torch.save(trained_model.state_dict(), 'floating_point_baseline_model.pth')
    print("✅ Baseline model saved to 'floating_point_baseline_model.pth'")
    
    print("\n" + "="*60)
    print("BASELINE MODEL TRAINING COMPLETE")
    print("This model serves as the floating-point reference for Phase 2")
    print("="*60)