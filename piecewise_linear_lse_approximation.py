"""
Piecewise Linear Approximation for Log-Sum-Exp
Based on Ada's research for CTC integer-only loss investigation
"""

import math

# Approximation parameters (based on Ada's error analysis)
Z_MIN = -10.0  # Minimum value for z = b - a (where a >= b)
Z_MAX = 0.0  # Maximum value for z = b - a
NUM_SEGMENTS = 40  # Number of piecewise linear segments

# Precomputed segment parameters for softplus(z) = log(1 + exp(z)) approximation
_SEGMENTS = [
    {'z_left': -10.000000, 'z_right': -9.750000, 
                  'slope': 0.000052, 'intercept': 0.000561}
    {'z_left': -9.750000, 'z_right': -9.500000, 
                  'slope': 0.000066, 'intercept': 0.000704}
    {'z_left': -9.500000, 'z_right': -9.250000, 
                  'slope': 0.000085, 'intercept': 0.000883}
    {'z_left': -9.250000, 'z_right': -9.000000, 
                  'slope': 0.000109, 'intercept': 0.001106}
    {'z_left': -9.000000, 'z_right': -8.750000, 
                  'slope': 0.000140, 'intercept': 0.001385}
    {'z_left': -8.750000, 'z_right': -8.500000, 
                  'slope': 0.000180, 'intercept': 0.001733}
    {'z_left': -8.500000, 'z_right': -8.250000, 
                  'slope': 0.000231, 'intercept': 0.002168}
    {'z_left': -8.250000, 'z_right': -8.000000, 
                  'slope': 0.000297, 'intercept': 0.002709}
    {'z_left': -8.000000, 'z_right': -7.750000, 
                  'slope': 0.000381, 'intercept': 0.003383}
    {'z_left': -7.750000, 'z_right': -7.500000, 
                  'slope': 0.000489, 'intercept': 0.004221}
    {'z_left': -7.500000, 'z_right': -7.250000, 
                  'slope': 0.000628, 'intercept': 0.005263}
    {'z_left': -7.250000, 'z_right': -7.000000, 
                  'slope': 0.000806, 'intercept': 0.006555}
    {'z_left': -7.000000, 'z_right': -6.750000, 
                  'slope': 0.001035, 'intercept': 0.008156}
    {'z_left': -6.750000, 'z_right': -6.500000, 
                  'slope': 0.001328, 'intercept': 0.010137}
    {'z_left': -6.500000, 'z_right': -6.250000, 
                  'slope': 0.001705, 'intercept': 0.012586}
    {'z_left': -6.250000, 'z_right': -6.000000, 
                  'slope': 0.002188, 'intercept': 0.015606}
    {'z_left': -6.000000, 'z_right': -5.750000, 
                  'slope': 0.002808, 'intercept': 0.019325}
    {'z_left': -5.750000, 'z_right': -5.500000, 
                  'slope': 0.003603, 'intercept': 0.023894}
    {'z_left': -5.500000, 'z_right': -5.250000, 
                  'slope': 0.004621, 'intercept': 0.029496}
    {'z_left': -5.250000, 'z_right': -5.000000, 
                  'slope': 0.005926, 'intercept': 0.036346}
    {'z_left': -5.000000, 'z_right': -4.750000, 
                  'slope': 0.007597, 'intercept': 0.044698}
    {'z_left': -4.750000, 'z_right': -4.500000, 
                  'slope': 0.009733, 'intercept': 0.054846}
    {'z_left': -4.500000, 'z_right': -4.250000, 
                  'slope': 0.012463, 'intercept': 0.067131}
    {'z_left': -4.250000, 'z_right': -4.000000, 
                  'slope': 0.015946, 'intercept': 0.081933}
    {'z_left': -4.000000, 'z_right': -3.750000, 
                  'slope': 0.020382, 'intercept': 0.099679}
    {'z_left': -3.750000, 'z_right': -3.500000, 
                  'slope': 0.026020, 'intercept': 0.120820}
    {'z_left': -3.500000, 'z_right': -3.250000, 
                  'slope': 0.033164, 'intercept': 0.145824}
    {'z_left': -3.250000, 'z_right': -3.000000, 
                  'slope': 0.042184, 'intercept': 0.175139}
    {'z_left': -3.000000, 'z_right': -2.750000, 
                  'slope': 0.053521, 'intercept': 0.209150}
    {'z_left': -2.750000, 'z_right': -2.500000, 
                  'slope': 0.067689, 'intercept': 0.248111}
    {'z_left': -2.500000, 'z_right': -2.250000, 
                  'slope': 0.085267, 'intercept': 0.292058}
    {'z_left': -2.250000, 'z_right': -2.000000, 
                  'slope': 0.106886, 'intercept': 0.340700}
    {'z_left': -2.000000, 'z_right': -1.750000, 
                  'slope': 0.133185, 'intercept': 0.393297}
    {'z_left': -1.750000, 'z_right': -1.500000, 
                  'slope': 0.164757, 'intercept': 0.448548}
    {'z_left': -1.500000, 'z_right': -1.250000, 
                  'slope': 0.202063, 'intercept': 0.504508}
    {'z_left': -1.250000, 'z_right': -1.000000, 
                  'slope': 0.245330, 'intercept': 0.558592}
    {'z_left': -1.000000, 'z_right': -0.750000, 
                  'slope': 0.294437, 'intercept': 0.607699}
    {'z_left': -0.750000, 'z_right': -0.500000, 
                  'slope': 0.348824, 'intercept': 0.648489}
    {'z_left': -0.500000, 'z_right': -0.250000, 
                  'slope': 0.407450, 'intercept': 0.677802}
    {'z_left': -0.250000, 'z_right': 0.000000, 
                  'slope': 0.468831, 'intercept': 0.693147}
]]

def log_add_exp_piecewise_approx(a, b):
    """
    Compute log(exp(a) + exp(b)) using piecewise linear approximation.
    More accurate than max-trick approximation while remaining integer-friendly.
    
    Args:
        a, b: Log values (can be any real numbers)
        
    Returns:
        Approximation of log(exp(a) + exp(b))
    """
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
    
    seg = _SEGMENTS[seg_idx]
    
    # Approximate log(1 + exp(z)) = slope * z + intercept
    log1p_exp_z_approx = seg['slope'] * z + seg['intercept']
    
    # Return a + log(1 + exp(b-a))
    return a + log1p_exp_z_approx

def log_sum_exp_piecewise_approx(log_vals):
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

def log_add_exp_exact(a, b):
    """Exact log-add-exp for comparison/testing."""
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))

def log_sum_exp_exact(log_vals):
    """Exact log-sum-exp for comparison/testing."""
    if not log_vals:
        return float('-inf')
    acc = log_vals[0]
    for v in log_vals[1:]:
        acc = log_add_exp_exact(acc, v)
    return acc

# Example usage and testing
if __name__ == "__main__":
    # Test with values similar to Ada's experiment
    import random
    
    def generate_log_probs(T, C=5, dist_type='uniform'):
        if dist_type == 'uniform':
            prob = 1.0 / C
            logp = math.log(prob)
            return [[logp for _ in range(C)] for _ in range(T)]
        elif dist_type == 'peaky':
            high = 0.9
            low = 0.1 / (C - 1)
            log_high = math.log(high)
            log_low = math.log(low)
            seq = []
            for t in range(T):
                high_idx = random.randrange(C)
                lps = [log_low] * C
                lps[high_idx] = log_high
                seq.append(lps)
            return seq
    
    print("Testing piecewise linear log-sum-exp approximation:")
    print("T   | Dist    | Exact LSE   | Approx LSE   | Abs Error   | Rel Error")
    print("----|---------|-------------|--------------|-------------|----------")
    
    sequence_lengths = [10, 50, 100, 200]
    dist_types = ['uniform', 'peaky']
    
    for T in sequence_lengths:
        for dist in dist_types:
            log_probs = generate_log_probs(T, C=5, dist_type=dist)
            flat = [lp for row in log_probs for lp in row]
            
            exact = log_sum_exp_exact(flat)
            approx = log_sum_exp_piecewise_approx(flat)
            
            abs_error = abs(approx - exact)
            rel_error = abs_error / abs(exact) if abs(exact) > 1e-10 else abs_error
            
            print(f"{T:2d}  | {dist:5s} | {exact:9.6f} | {approx:10.6f} | {abs_error:9.6f} | {rel_error:8.6f}".format(
                T=T, dist=dist, exact=exact, approx=approx, abs_error=abs_error, rel_error=rel_error))
