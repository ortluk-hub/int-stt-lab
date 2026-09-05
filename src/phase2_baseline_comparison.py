#!/usr/bin/env python3
"""
Phase 2 Baseline Comparison Script
Compares integer-only STT model vs floating-point baseline model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from floating_point_baseline import FloatingPointSpeechCommandNet
import sys; sys.path.insert(0, 'phase1/prototype'); from simple_stt_model import SimpleSpeechCommandNet
import sys; sys.path.insert(0, 'src'); from train_baseline import create_dummy_dataset
import sys; sys.path.insert(0, 'src'); from niti_integration_training import train_integer_only


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_model(model, dataloader, model_type="unknown"):
    """Evaluate model on given dataset"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(dataloader)
    
    print(f"{model_type} Model Evaluation:")
    print(f"  Accuracy: {accuracy:.2f}%")
    print(f"  Average Loss: {avg_loss:.4f}")
    print(f"  Parameters: {count_parameters(model)}")
    
    return accuracy, avg_loss


def compare_models():
    """Compare integer-only and floating-point models"""
    print("=" * 60)
    print("PHASE 2 BASELINE COMPARISON")
    print("Integer-only STT Model vs Floating-point Baseline")
    print("=" * 60)
    
    # Create evaluation dataset
    print("\nCreating evaluation dataset...")
    eval_loader = create_dummy_dataset(batch_size=32, num_samples=200)
    print(f"Evaluation dataset: {len(eval_loader.dataset)} samples")
    
    # Test Floating-point Baseline
    print("\n" + "-" * 50)
    print("TESTING FLOATING-POINT BASELINE MODEL")
    print("-" * 50)
    
    # Load or create floating-point baseline model
    baseline_model = FloatingPointSpeechCommandNet(input_size=40, hidden_size=64, num_classes=5)
    baseline_path = 'floating_point_baseline_model.pth'
    
    if os.path.exists(baseline_path):
        print(f"Loading pre-trained baseline from {baseline_path}")
        baseline_model.load_state_dict(torch.load(baseline_path))
    else:
        print("Training floating-point baseline model...")
        # Quick training for comparison
        train_loader = create_dummy_dataset(batch_size=32, num_samples=100)
        baseline_model = train_baseline(baseline_model, train_loader, num_epochs=5)
        torch.save(baseline_model.state_dict(), baseline_path)
        print(f"Saved baseline model to {baseline_path}")
    
    baseline_acc, baseline_loss = evaluate_model(baseline_model, eval_loader, "Floating-point Baseline")
    
    # Test Integer-only Model (simulated)
    print("\n" + "-" * 50)
    print("TESTING INTEGER-ONLY MODEL (SIMULATED)")
    print("-" * 50)
    
    integer_model = SimpleSpeechCommandNet(input_size=40, hidden_size=64, num_classes=5)
    
    # Quick training simulation for integer model
    print("Training integer-only model (simulation)...")
    # We'll use the same training data but with integer simulation
    train_loader = create_dummy_dataset(batch_size=32, num_samples=100)
    
    # Simple training loop for integer model (simulated)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(integer_model.parameters(), lr=0.01, momentum=0.9)
    
    integer_model.train()
    for epoch in range(3):  # Fewer epochs for demo
        running_loss = 0.0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            
            # Simulate integer-only forward pass (quantize weights/activations)
            # This is a simplified simulation - in practice would use NITI framework
            def quantize_tensor(tensor, scale=127.0):
                quantized = (tensor * scale).round().clamp(-128, 127)
                return quantized / scale
            
            # Apply quantization simulation to linear layers
            with torch.no_grad():
                for layer in [integer_model.fc1, integer_model.fc2, integer_model.fc3]:
                    if hasattr(layer, 'linear'):
                        layer.linear.weight.data = quantize_tensor(layer.linear.weight.data)
                        if layer.linear.bias is not None:
                            layer.linear.bias.data = quantize_tensor(layer.linear.bias.data)
            
            outputs = integer_model(inputs)
            # Quantize outputs
            outputs = quantize_tensor(outputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        print(f'  Integer Epoch [{epoch+1}/3] Loss: {running_loss/len(train_loader):.4f}')
    
    integer_acc, integer_loss = evaluate_model(integer_model, eval_loader, "Integer-only (Simulated)")
    
    # Summary
    print("\n" + "=" * 60)
    print("PHASE 2 BASELINE COMPARISON RESULTS")
    print("=" * 60)
    print(f"{'Model Type':<25} {'Accuracy':<12} {'Loss':<12} {'Parameters'}")
    print("-" * 60)
    print(f"{'Floating-point Baseline':<25} {baseline_acc:<12.2f} {baseline_loss:<12.4f} {count_parameters(baseline_model)}")
    print(f"{'Integer-only (Simulated)':<25} {integer_acc:<12.2f} {integer_loss:<12.4f} {count_parameters(integer_model)}")
    print("-" * 60)
    
    acc_diff = baseline_acc - integer_acc
    loss_diff = integer_loss - baseline_loss  # Positive means integer has higher loss
    
    print(f"\nDifferences (Baseline - Integer):")
    print(f"  Accuracy Difference: {acc_diff:+.2f}%")
    print(f"  Loss Difference: {loss_diff:+.4f}")
    
    if abs(acc_diff) < 5.0:  # Within 5% accuracy
        print("✓ Integer-only model performance is comparable to baseline")
    else:
        print("⚠ Integer-only model shows significant performance gap")
    
    print("\n" + "=" * 60)
    print("PHASE 2 BASELINE ESTABLISHED")
    print("Ready to proceed with numerical validation and documentation")
    print("=" * 60)
    
    # Save results for documentation
    results = {
        'baseline_accuracy': baseline_acc,
        'baseline_loss': baseline_loss,
        'baseline_parameters': count_parameters(baseline_model),
        'integer_accuracy': integer_acc,
        'integer_loss': integer_loss,
        'integer_parameters': count_parameters(integer_model),
        'accuracy_difference': acc_diff,
        'loss_difference': loss_diff
    }
    
    import json
    with open('phase2_baseline_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("✓ Results saved to 'phase2_baseline_results.json'")
    
    return results


if __name__ == "__main__":
    compare_models()