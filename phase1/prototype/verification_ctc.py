#!/home/ortluk/ortluk-hub/int-stt-lab/venv/bin/python
"""
Verify CTC loss approximation error is below 1e-3 target.
"""
import sys
sys.path.insert(0, '/home/ortluk/ortluk-hub/int-stt-lab')
import torch
import torch.nn.functional as F
import numpy as np
from ctc_loss_implementation import ctc_loss_forward_approximated, ctc_loss_forward_exact

def test_ctc_approximation():
    print("Testing CTC loss approximation error...")
    
    # Test case 1: Simple sequence
    log_probs = [
        [torch.log(torch.tensor(0.8)), torch.log(torch.tensor(0.1)), torch.log(torch.tensor(0.1))],
        [torch.log(torch.tensor(0.1)), torch.log(torch.tensor(0.8)), torch.log(torch.tensor(0.1))]
    ]
    targets = [[1]]
    input_lengths = [2]
    target_lengths = [1]
    
    approx_loss = ctc_loss_forward_approximated(log_probs, targets, input_lengths, target_lengths)
    exact_loss = ctc_loss_forward_exact(log_probs, targets, input_lengths, target_lengths)
    error = abs(approx_loss - exact_loss)
    print(f"Test 1 - Approx: {approx_loss:.6f}, Exact: {exact_loss:.6f}, Error: {error:.6f}")
    
    # Test case 2: Longer sequence
    log_probs2 = []
    for t in range(5):
        # Random log probabilities (normalized)
        probs = torch.softmax(torch.randn(3), dim=0)
        log_probs2.append([torch.log(p) for p in probs])
    targets2 = [[1, 2]]
    input_lengths2 = [5]
    target_lengths2 = [2]
    
    approx_loss2 = ctc_loss_forward_approximated(log_probs2, targets2, input_lengths2, target_lengths2)
    exact_loss2 = ctc_loss_forward_exact(log_probs2, targets2, input_lengths2, target_lengths2)
    error2 = abs(approx_loss2 - exact_loss2)
    print(f"Test 2 - Approx: {approx_loss2:.6f}, Exact: {exact_loss2:.6f}, Error: {error2:.6f}")
    
    # Test case 3: Batch size > 1
    log_probs3 = [
        [torch.log(torch.tensor(0.7)), torch.log(torch.tensor(0.2)), torch.log(torch.tensor(0.1))],
        [torch.log(torch.tensor(0.2)), torch.log(torch.tensor(0.6)), torch.log(torch.tensor(0.2))],
        [torch.log(torch.tensor(0.1)), torch.log(torch.tensor(0.1)), torch.log(torch.tensor(0.8))]
    ]
    targets3 = [[0], [2]]
    input_lengths3 = [3, 3]
    target_lengths3 = [1, 1]
    
    approx_loss3 = ctc_loss_forward_approximated(log_probs3, targets3, input_lengths3, target_lengths3)
    exact_loss3 = ctc_loss_forward_exact(log_probs3, targets3, input_lengths3, target_lengths3)
    error3 = abs(approx_loss3 - exact_loss3)
    print(f"Test 3 - Approx: {approx_loss3:.6f}, Exact: {exact_loss3:.6f}, Error: {error3:.6f}")
    
    max_error = max(error, error2, error3)
    print(f"\nMaximum approximation error: {max_error:.6f}")
    if max_error < 1e-3:
        print("✅ PASS: CTC approximation error is below 1e-3 target")
        return True
    else:
        print("❌ FAIL: CTC approximation error exceeds 1e-3 target")
        return False

if __name__ == "__main__":
    test_ctc_approximation()
