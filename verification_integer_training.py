#!/usr/bin/env python3
"""
Verification script for integer-only training pipeline
Tests the NITI integer-only training implementation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase1', 'prototype'))

def test_niti_integer_training_import():
    """Test that the NITI integer training module can be imported"""
    try:
        from src.niti_integer_training import train_integer_only, verify_ctc_approximation_in_training
        print("✅ Successfully imported niti_integer_training module")
        return True
    except Exception as e:
        print(f"❌ Failed to import niti_integer_training: {e}")
        return False

def test_ctc_approximation_verification():
    """Test the CTC approximation verification function"""
    try:
        from src.niti_integer_training import verify_ctc_approximation_in_training
        result = verify_ctc_approximation_in_training()
        if result:
            print("✅ CTC approximation verification passed")
            return True
        else:
            print("❌ CTC approximation verification failed")
            return False
    except Exception as e:
        print(f"❌ Error during CTC approximation verification: {e}")
        return False

def test_training_simulation():
    """Test that the training simulation runs without error"""
    try:
        from src.niti_integer_training import train_integer_only
        import torch
        from simple_stt_model import SimpleSpeechCommandNet
        from torch.utils.data import DataLoader, TensorDataset
        
        # Create a minimal model and data for testing
        model = SimpleSpeechCommandNet(input_size=40, hidden_size=32, num_classes=5)
        
        # Create tiny dataset
        inputs = torch.randn(32, 40)
        labels = torch.randint(0, 5, (32,))
        dataset = TensorDataset(inputs, labels)
        dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
        
        # Run training for just 1 epoch
        train_integer_only(model, dataloader, num_epochs=1)
        print("✅ Training simulation completed successfully")
        return True
    except Exception as e:
        print(f"❌ Error during training simulation: {e}")
        return False

def main():
    """Run all verification tests"""
    print("=== Integer-Only Training Pipeline Verification ===\n")
    
    tests = [
        ("Import Test", test_niti_integer_training_import),
        ("CTC Approximation Verification", test_ctc_approximation_verification),
        ("Training Simulation", test_training_simulation)
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"Running {name}...")
        if test_func():
            passed += 1
        print()
    
    print(f"=== Results: {passed}/{total} tests passed ===")
    
    if passed == total:
        print("🎉 All verification tests passed!")
        return True
    else:
        print("❌ Some verification tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)