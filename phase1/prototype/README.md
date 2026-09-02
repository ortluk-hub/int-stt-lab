# Phase 1 Prototype: Integer-Only STT

This directory contains a prototype implementation for Phase 1 of the 
integer-arctic CTC speech-to-text project.

## Contents

- `simple_stt_model.py`: A simple feed-forward network for spectrogram patches
- `train_integer_simulation.py`: Training loop that simulates integer-only training principles
- `verify_integer_principles.py`: Basic verification of integer-only concepts (in parent directory)

## Next Steps

1. **Framework Integration**: Replace the PyTorch model with NITI/NITRO-D equivalent
2. **Actual Integer-Only Training**: Use the frameworks for real integer-only operations
3. **CTC Loss Investigation**: Implement integer-only CTC loss function
4. **Speech Commands Dataset**: Replace dummy data with actual Speech Commands dataset
5. **QNN Preparation**: Prepare model for ONNX export and QNN deployment verification

## How to Run

```bash
# Activate the QNN virtual environment (if available)
source /home/ortluk/src/executorch/.venv-qnn/bin/activate

# Run the verification
python3 verify_integer_principles.py

# Run the prototype training simulation
python3 prototype/train_integer_simulation.py
```

## Notes

- The current implementation uses PyTorch for demonstration and simulates integer-only principles.
- For actual integer-only training, integrate with NITI or NITRO-D frameworks.
- See the research documentation in `/home/ortluk/research/` for framework references and CTC investigation.
