# NITI Framework Integration into Integer-Only Training Pipeline

## Overview
This document describes the integration of the NITI framework components into the integer-only training pipeline for speech-to-text (STT) using simulated operations as a precursor to actual CUDA kernel integration.

## Objectives
- Replace simulated integer-only operations with actual NITI framework layers (TiLinear, TiReLU, etc.)
- Ensure compatibility with PyTorch's CTCLoss for differentiability
- Verify that integer-only operations are used (weights and activations in int8 format during forward pass)
- Update verification scripts to test the NITI-integrated training pipeline

## Changes Made

### 1. Simulated NITI Layers (`src/niti_layers_simulated.py`)
- Created differentiable simulations of `TiLinear`, `TiReLU`, and `TiDropout` layers
- Used straight-through estimator for quantization operations to preserve gradients
- Simulated exponent handling as scalar float tensors
- Matrix multiplication simulation includes padding to multiples of 4 as in the original NITI framework

### 2. Updated Training Pipeline (`src/niti_training.py`)
- Replaced `IntegerOnlyLinear` and `IntegerOnlyFramewiseNet` with simulated NITI layers
- Maintained the same training loop structure using PyTorch's CTCLoss
- Input to NITI layers is a tuple of (activation tensor, exponent tensor)
- Output logits are converted to float for CTCLoss (using simplified exponent handling in simulation)

### 3. Verification Scripts (`verification_integer_training.py`)
- Updated to import and test the new NITI-integrated training pipeline
- Tests include:
  - Import test: verifies module can be imported
  - Model creation test: checks model instantiation and parameter count
  - Forward pass test: validates tensor shapes through the network
  - Training simulation test: runs a short training loop to ensure no errors

## Implementation Details

### Differentiable Quantization
To simulate int8 operations while preserving gradients for backpropagation, we used a straight-through estimator:
```python
def quantize_tensor(tensor, scale=127.0):
    scaled = tensor * scale
    rounded = torch.round(scaled) + (scaled - scaled.detach())
    clamped = rounded.clamp(-128, 127)
    dequantized = clamped / scale
    return dequantized
```

### Layer Simulation
Each NITI layer receives and returns a tuple `(tensor, exponent)` where:
- `tensor`: float tensor simulating the int8 values (in range [-1,1] after dequantization)
- `exponent`: float tensor representing the scaling exponent (simulated as scalar for simplicity)

The `TiLinear` layer performs:
1. Weight quantization using the same straight-through estimator
2. Matrix multiplication via `int8mm_simulated` (which applies quantization and padding)
3. Exponent calculation using simulated `act_calc` function
4. Returns activated tensor and updated exponent

### Training Loop
The training loop remains unchanged from a high-level perspective:
- Forward pass through NITI layers to get logits
- Log softmax to get log probabilities
- CTCLoss computation
- Backward pass and optimizer step

## Verification Results
All verification tests pass:
1. **Import Test**: ✅ Successfully imported niti_training module
2. **Model Creation Test**: ✅ Model created successfully with 6976 parameters
3. **Forward Pass Test**: ✅ Forward pass successful, output shape: torch.Size([2, 10, 5])
4. **Training Simulation Test**: ✅ Training simulation completed successfully

## Next Steps
1. Build NITI CUDA extensions (cutlassconv_cuda, int8mm_cuda, etc.)
2. Replace simulated operations with actual NITI kernels
3. Verify training works with the real NITI framework
4. Document performance and numerical differences between simulation and actual kernels

## Dependencies
- PyTorch
- CUDA toolkit (for future extension build)
- NITI framework source (included in phase1/niti/)

## Notes
The current implementation uses simulated NITI operations due to missing CUDA extensions. The simulation is designed to closely mimic the behavior of the actual NITI framework while being differentiable and runnable on CPU without special hardware.