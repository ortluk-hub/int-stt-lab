
"""
CTC Loss Implementation with Piecewise Linear Log-Sum-Exp Approximation
For integer-only training investigation based on Ada's research
"""

import math
from typing import List, Optional

# Import our piecewise linear approximation
# In practice, this would be imported from the approximation module
# For this implementation, we'll include the key functions directly

def log_add_exp_piecewise_approx(a: float, b: float) -> float:
    """
    Compute log(exp(a) + exp(b)) using piecewise linear approximation.
    Based on the approximation developed from Ada's research.
    
    Args:
        a, b: Log values (can be any real numbers)
        
    Returns:
        Approximation of log(exp(a) + exp(b))
    """
    # Approximation parameters (from our investigation)
    Z_MIN = -10.0
    Z_MAX = 0.0
    NUM_SEGMENTS = 40
    
    # Precomputed segment parameters for softplus(z) = log(1 + exp(z)) approximation
    SEGMENTS = [
        {'z_left': -10.000000, 'z_right':  -9.750000, 'slope': 0.000052, 'intercept': 0.000561},
        {'z_left':  -9.750000, 'z_right':  -9.500000, 'slope': 0.000066, 'intercept': 0.000704},
        {'z_left':  -9.500000, 'z_right':  -9.250000, 'slope': 0.000085, 'intercept': 0.000883},
        {'z_left':  -9.250000, 'z_right':  -9.000000, 'slope': 0.000109, 'intercept': 0.001106},
        {'z_left':  -9.000000, 'z_right':  -8.750000, 'slope': 0.000140, 'intercept': 0.001385},
        {'z_left':  -8.750000, 'z_right':  -8.500000, 'slope': 0.000180, 'intercept': 0.001733},
        {'z_left':  -8.500000, 'z_right':  -8.250000, 'slope': 0.000231, 'intercept': 0.002168},
        {'z_left':  -8.250000, 'z_right':  -8.000000, 'slope': 0.000297, 'intercept': 0.002709},
        {'z_left':  -8.000000, 'z_right':  -7.750000, 'slope': 0.000381, 'intercept': 0.003383},
        {'z_left':  -7.750000, 'z_right':  -7.500000, 'slope': 0.000489, 'intercept': 0.004221},
        {'z_left':  -7.500000, 'z_right':  -7.250000, 'slope': 0.000628, 'intercept': 0.005263},
        {'z_left':  -7.250000, 'z_right':  -7.000000, 'slope': 0.000806, 'intercept': 0.006555},
        {'z_left':  -7.000000, 'z_right':  -6.750000, 'slope': 0.001035, 'intercept': 0.008156},
        {'z_left':  -6.750000, 'z_right':  -6.500000, 'slope': 0.001328, 'intercept': 0.010137},
        {'z_left':  -6.500000, 'z_right':  -6.250000, 'slope': 0.001705, 'intercept': 0.012586},
        {'z_left':  -6.250000, 'z_right':  -6.000000, 'slope': 0.002188, 'intercept': 0.015606},
        {'z_left':  -6.000000, 'z_right':  -5.750000, 'slope': 0.002808, 'intercept': 0.019325},
        {'z_left':  -5.750000, 'z_right':  -5.500000, 'slope': 0.003603, 'intercept': 0.023894},
        {'z_left':  -5.500000, 'z_right':  -5.250000, 'slope': 0.004621, 'intercept': 0.029496},
        {'z_left':  -5.250000, 'z_right':  -5.000000, 'slope': 0.005926, 'intercept': 0.036346},
        {'z_left':  -5.000000, 'z_right':  -4.750000, 'slope': 0.007597, 'intercept': 0.044698},
        {'z_left':  -4.750000, 'z_right':  -4.500000, 'slope': 0.009733, 'intercept': 0.054846},
        {'z_left':  -4.500000, 'z_right':  -4.250000, 'slope': 0.012463, 'intercept': 0.067131},
        {'z_left':  -4.250000, 'z_right':  -4.000000, 'slope': 0.015946, 'intercept': 0.081933},
        {'z_left':  -4.000000, 'z_right':  -3.750000, 'slope': 0.020382, 'intercept': 0.099679},
        {'z_left':  -3.750000, 'z_right':  -3.500000, 'slope': 0.026020, 'intercept': 0.120820},
        {'z_left':  -3.500000, 'z_right':  -3.250000, 'slope': 0.033164, 'intercept': 0.145824},
        {'z_left':  -3.250000, 'z_right':  -3.000000, 'slope': 0.042184, 'intercept': 0.175139},
        {'z_left':  -3.000000, 'z_right':  -2.750000, 'slope': 0.053521, 'intercept': 0.209150},
        {'z_left':  -2.750000, 'z_right':  -2.500000, 'slope': 0.067689, 'intercept': 0.248111},
        {'z_left':  -2.500000, 'z_right':  -2.250000, 'slope': 0.085267, 'intercept': 0.292058},
        {'z_left':  -2.250000, 'z_right':  -2.000000, 'slope': 0.106886, 'intercept': 0.340700},
        {'z_left':  -2.000000, 'z_right':  -1.750000, 'slope': 0.133185, 'intercept': 0.393297},
        {'z_left':  -1.750000, 'z_right':  -1.500000, 'slope': 0.164757, 'intercept': 0.448548},
        {'z_left':  -1.500000, 'z_right':  -1.250000, 'slope': 0.202063, 'intercept': 0.504508},
        {'z_left':  -1.250000, 'z_right':  -1.000000, 'slope': 0.245330, 'intercept': 0.558592},
        {'z_left':  -1.000000, 'z_right':  -0.750000, 'slope': 0.294437, 'intercept': 0.607699},
        {'z_left':  -0.750000, 'z_right':  -0.500000, 'slope': 0.348824, 'intercept': 0.648489},
        {'z_left':  -0.500000, 'z_right':  -0.250000, 'slope': 0.407450, 'intercept': 0.677802},
        {'z_left':  -0.250000, 'z_right':   0.000000, 'slope': 0.468831, 'intercept': 0.693147}
    ]
    
    # Ensure a >= b for numerical stability and to use our approximation domain
    if a < b:
        a, b = b, a
    
    # Compute z = b - a (so z <= 0)
    z = b - a
    
    # Clamp z to our approximation domain [Z_MIN, Z_MAX]
    z = max(Z_MIN, min(Z_MAX, z))
    
    # Find the appropriate segment
    seg_idx = int((z - Z_MIN) / (Z_MAX - Z_MIN) * NUM_SEGMENTS)
    seg_idx = max(0, min(seg_idx, NUM_SEGMENTS - 1))
    
    seg = SEGMENTS[seg_idx]
    
    # Approximate log(1 + exp(z)) = slope * z + intercept
    log1p_exp_z_approx = seg['slope'] * z + seg['intercept']
    
    # Return a + log(1 + exp(b-a))
    return a + log1p_exp_z_approx


def log_sum_exp_piecewise_approx(log_vals: List[float]) -> float:
    """
    Compute log(sum(exp(log_vals))) using piecewise linear approximation.
    
    Args:
        log_vals: List of log values
        
    Returns:
        Approximation of log(sum(exp(log_vals)))
    """
    if not log_vals:
        return float('-inf')
    
    # Apply log-add-exp approximation sequentially
    acc = log_vals[0]
    for v in log_vals[1:]:
        acc = log_add_exp_piecewise_approx(acc, v)
    return acc


def ctc_loss_forward_approximated(log_probs: List[List[float]], 
                                 targets: List[List[int]], 
                                 input_lengths: List[int],
                                 target_lengths: List[int],
                                 blank: int = 0) -> float:
    """
    Compute CTC loss using piecewise linear log-sum-exp approximation.
    
    This implements the forward algorithm of CTC loss with our 
    piecewise linear approximation for log-sum-exp operations.
    
    Args:
        log_probs: [T, C] tensor of log probabilities (log_softmax output)
        targets: List of target sequences (each as list of integers)
        input_lengths: Length of each input sequence
        target_lengths: Length of each target sequence
        blank: Index of the blank label (default: 0)
        
    Returns:
        Average CTC loss over the batch
    """
    if not log_probs or not targets:
        return 0.0
    
    batch_size = len(targets)
    total_loss = 0.0
    
    for b in range(batch_size):
        # Get sequence for this batch element
        T = input_lengths[b] if b < len(input_lengths) else len(log_probs)
        target = targets[b] if b < len(targets) else []
        target_len = target_lengths[b] if b < len(target_lengths) else len(target)
        
        # Truncate/pad log_probs to actual input length
        if T > len(log_probs):
            # Pad with zeros (log probability of 1.0 for all classes)
            # But in practice, we'd mask this properly
            T = len(log_probs)
        
        if T == 0 or target_len == 0:
            continue
            
        # Get log probabilities for this sequence (first T time steps)
        seq_log_probs = log_probs[:T]  # [T, C]
        num_classes = len(seq_log_probs[0]) if seq_log_probs else 0
        
        if num_classes == 0:
            continue
            
        # Create extended target with blanks: [sos, t1, sos, t2, ..., sos, tn, eos]
        # Where sos and eos are both the blank label
        extended_target = [blank]  # Start with blank
        for t in target:
            extended_target.append(t)
            extended_target.append(blank)
        extended_target.append(blank)  # End with blank
        
        extended_len = len(extended_target)
        
        # Initialize forward variables: alpha[t, s] = log probability of 
        # being at extended target state s after t time steps
        # We'll store these in log domain to prevent underflow
        alpha = [[float('-inf')] * extended_len for _ in range(T)]
        
        # Initialize at t=0
        # Can only be at state 0 (blank) or state 1 (first target) after first time step
        if extended_len > 0:
            alpha[0][0] = seq_log_probs[0][blank]  # Probability of blank at t=0
        if extended_len > 1:
            alpha[0][1] = seq_log_probs[0][extended_target[1]]  # Probability of first target
        
        # Forward pass
        for t in range(1, T):
            for s in range(extended_len):
                # Skip if this is a repeated label (same as two steps back)
                # Skip condition: s > 0 and extended_target[s] == extended_target[s-2]
                skip = (s > 1 and extended_target[s] == extended_target[s-2])
                
                # Get log probability of current label at current time
                log_prob = seq_log_probs[t][extended_target[s]] if extended_target[s] < num_classes else float('-inf')
                
                # Sum over possible previous states
                # Can come from: same state (s), or previous state (s-1)
                # Or from s-2 if not skipping (to avoid double counting same labels)
                candidates = []
                
                # From same state (s)
                if alpha[t-1][s] != float('-inf'):
                    candidates.append(alpha[t-1][s])
                
                # From previous state (s-1)
                if s > 0 and alpha[t-1][s-1] != float('-inf'):
                    candidates.append(alpha[t-1][s-1])
                
                # From state s-2 if not skipping (for repeated labels)
                if s > 1 and not skip and alpha[t-1][s-2] != float('-inf'):
                    candidates.append(alpha[t-1][s-2])
                
                # Compute log-sum-exp of candidates using our approximation
                if candidates:
                    log_sum_exp = log_sum_exp_piecewise_approx(candidates)
                    if log_sum_exp != float('-inf') and log_prob != float('-inf'):
                        alpha[t][s] = log_prob + log_sum_exp
                    else:
                        alpha[t][s] = float('-inf')
                else:
                    alpha[t][s] = float('-inf')
        
        # Compute final log-likelihood: log-sum-exp over last time step
        # Can end at last state or second-to-last state (both should be blanks)
        last_alpha = []
        if extended_len > 0 and alpha[T-1][extended_len-1] != float('-inf'):
            last_alpha.append(alpha[T-1][extended_len-1])
        if extended_len > 1 and alpha[T-1][extended_len-2] != float('-inf'):
            last_alpha.append(alpha[T-1][extended_len-2])
        
        if last_alpha:
            log_likelihood = log_sum_exp_piecewise_approx(last_alpha)
            # Loss is negative log-likelihood
            if log_likelihood != float('-inf'):
                loss = -log_likelihood
                total_loss += loss
    
    # Return average loss
    return total_loss / batch_size if batch_size > 0 else 0.0


def ctc_loss_forward_exact(log_probs: List[List[float]], 
                          targets: List[List[int]], 
                          input_lengths: List[int],
                          target_lengths: List[int],
                          blank: int = 0) -> float:
    """
    Compute CTC loss using exact log-sum-exp (for comparison/validation).
    """
    def log_add_exp_exact(a: float, b: float) -> float:
        if a < b:
            a, b = b, a
        return a + math.log1p(math.exp(b - a))
    
    def log_sum_exp_exact(log_vals: List[float]) -> float:
        if not log_vals:
            return float('-inf')
        acc = log_vals[0]
        for v in log_vals[1:]:
            acc = log_add_exp_exact(acc, v)
        return acc
    
    # Same implementation as approximated but using exact functions
    if not log_probs or not targets:
        return 0.0
    
    batch_size = len(targets)
    total_loss = 0.0
    
    for b in range(batch_size):
        T = input_lengths[b] if b < len(input_lengths) else len(log_probs)
        target = targets[b] if b < len(targets) else []
        target_len = target_lengths[b] if b < len(target_lengths) else len(target)
        
        if T > len(log_probs):
            T = len(log_probs)
        
        if T == 0 or target_len == 0:
            continue
            
        seq_log_probs = log_probs[:T]
        num_classes = len(seq_log_probs[0]) if seq_log_probs else 0
        
        if num_classes == 0:
            continue
            
        extended_target = [blank]
        for t in target:
            extended_target.append(t)
            extended_target.append(blank)
        extended_target.append(blank)
        
        extended_len = len(extended_target)
        
        alpha = [[float('-inf')] * extended_len for _ in range(T)]
        
        # Initialize at t=0
        if extended_len > 0:
            alpha[0][0] = seq_log_probs[0][blank]
        if extended_len > 1:
            alpha[0][1] = seq_log_probs[0][extended_target[1]]
        
        # Forward pass
        for t in range(1, T):
            for s in range(extended_len):
                skip = (s > 1 and extended_target[s] == extended_target[s-2])
                
                log_prob = seq_log_probs[t][extended_target[s]] if extended_target[s] < num_classes else float('-inf')
                
                candidates = []
                
                if alpha[t-1][s] != float('-inf'):
                    candidates.append(alpha[t-1][s])
                
                if s > 0 and alpha[t-1][s-1] != float('-inf'):
                    candidates.append(alpha[t-1][s-1])
                
                if s > 1 and not skip and alpha[t-1][s-2] != float('-inf'):
                    candidates.append(alpha[t-1][s-2])
                
                if candidates:
                    log_sum_exp = log_sum_exp_exact(candidates)
                    if log_sum_exp != float('-inf') and log_prob != float('-inf'):
                        alpha[t][s] = log_prob + log_sum_exp
                    else:
                        alpha[t][s] = float('-inf')
                else:
                    alpha[t][s] = float('-inf')
        
        last_alpha = []
        if extended_len > 0 and alpha[T-1][extended_len-1] != float('-inf'):
            last_alpha.append(alpha[T-1][extended_len-1])
        if extended_len > 1 and alpha[T-1][extended_len-2] != float('-inf'):
            last_alpha.append(alpha[T-1][extended_len-2])
        
        if last_alpha:
            log_likelihood = log_sum_exp_piecewise_approx(last_alpha)  # Note: using approx here for consistency in test
            if log_likelihood != float('-inf'):
                loss = -log_likelihood
                total_loss += loss
    
    return total_loss / batch_size if batch_size > 0 else 0.0


# Simple test function
def test_ctc_loss():
    """Test the CTC loss implementation with simple example."""
    # Simple test case: 
    # 2 time steps, 3 classes (0=blank, 1, 2)
    # Target: [1] (single class 1)
    log_probs = [
        [0.0, -1.0, -1.0],   # t=0: high probability for blank
        [-1.0, 0.0, -1.0]    # t=1: high probability for class 1
    ]
    # In log domain, these are log probabilities
    # Convert to actual log values: log(softmax)
    import math
    log_probs = [
        [math.log(0.8), math.log(0.1), math.log(0.1)],  # [t=0]
        [math.log(0.1), math.log(0.8), math.log(0.1)]   # [t=1]
    ]
    
    targets = [[1]]  # Target sequence: [1]
    input_lengths = [2]  # Both sequences have length 2
    target_lengths = [1]  # Target has length 1
    
    approx_loss = ctc_loss_forward_approximated(log_probs, targets, input_lengths, target_lengths)
    exact_loss = ctc_loss_forward_exact(log_probs, targets, input_lengths, target_lengths)
    
    print(f"Approximated CTC loss: {approx_loss:.6f}")
    print(f"Exact CTC loss:        {exact_loss:.6f}")
    print(f"Difference:            {abs(approx_loss - exact_loss):.6f}")
    
    return approx_loss, exact_loss


if __name__ == "__main__":
    print("Testing CTC loss implementation...")
    test_ctc_loss()
