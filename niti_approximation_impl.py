"""
NITI-Specific Implementation of Piecewise Linear Log-Sum-Exp Approximation

Implements the approximation for use within NITI's int8 matrix operations
and block scaling scheme, based on research findings.

"""

from typing import Tuple, List

_SEGMENTS_NITI = [
    {'z_left': -10.000000, 'z_right': -9.750000, 'slope': 0.000052, 'intercept': 0.000561},
    {'z_left': -9.750000, 'z_right': -9.500000, 'slope': 0.000066, 'intercept': 0.000704},
    {'z_left': -9.500000, 'z_right': -9.250000, 'slope': 0.000085, 'intercept': 0.000883},
    {'z_left': -9.250000, 'z_right': -9.000000, 'slope': 0.000109, 'intercept': 0.001106},
    {'z_left': -9.000000, 'z_right': -8.750000, 'slope': 0.000140, 'intercept': 0.001385},
    {'z_left': -8.750000, 'z_right': -8.500000, 'slope': 0.000180, 'intercept': 0.001733},
    {'z_left': -8.500000, 'z_right': -8.250000, 'slope': 0.000231, 'intercept': 0.002168},
    {'z_left': -8.250000, 'z_right': -8.000000, 'slope': 0.000297, 'intercept': 0.002709},
    {'z_left': -8.000000, 'z_right': -7.750000, 'slope': 0.000381, 'intercept': 0.003383},
    {'z_left': -7.750000, 'z_right': -7.500000, 'slope': 0.000489, 'intercept': 0.004221},
    {'z_left': -7.500000, 'z_right': -7.250000, 'slope': 0.000628, 'intercept': 0.005263},
    {'z_left': -7.250000, 'z_right': -7.000000, 'slope': 0.000806, 'intercept': 0.006555},
    {'z_left': -7.000000, 'z_right': -6.750000, 'slope': 0.001035, 'intercept': 0.008156},
    {'z_left': -6.750000, 'z_right': -6.500000, 'slope': 0.001328, 'intercept': 0.010137},
    {'z_left': -6.500000, 'z_right': -6.250000, 'slope': 0.001705, 'intercept': 0.012586},
    {'z_left': -6.250000, 'z_right': -6.000000, 'slope': 0.002188, 'intercept': 0.015606},
    {'z_left': -6.000000, 'z_right': -5.750000, 'slope': 0.002808, 'intercept': 0.019325},
    {'z_left': -5.750000, 'z_right': -5.500000, 'slope': 0.003603, 'intercept': 0.023894},
    {'z_left': -5.500000, 'z_right': -5.250000, 'slope': 0.004621, 'intercept': 0.029496},
    {'z_left': -5.250000, 'z_right': -5.000000, 'slope': 0.005926, 'intercept': 0.036346},
    {'z_left': -5.000000, 'z_right': -4.750000, 'slope': 0.007597, 'intercept': 0.044698},
    {'z_left': -4.750000, 'z_right': -4.500000, 'slope': 0.009733, 'intercept': 0.054846},
    {'z_left': -4.500000, 'z_right': -4.250000, 'slope': 0.012463, 'intercept': 0.067131},
    {'z_left': -4.250000, 'z_right': -4.000000, 'slope': 0.015946, 'intercept': 0.081933},
    {'z_left': -4.000000, 'z_right': -3.750000, 'slope': 0.020382, 'intercept': 0.099679},
    {'z_left': -3.750000, 'z_right': -3.500000, 'slope': 0.026020, 'intercept': 0.120820},
    {'z_left': -3.500000, 'z_right': -3.250000, 'slope': 0.033164, 'intercept': 0.145824},
    {'z_left': -3.250000, 'z_right': -3.000000, 'slope': 0.042184, 'intercept': 0.175139},
    {'z_left': -3.000000, 'z_right': -2.750000, 'slope': 0.053521, 'intercept': 0.209150},
    {'z_left': -2.750000, 'z_right': -2.500000, 'slope': 0.067689, 'intercept': 0.248111},
    {'z_left': -2.500000, 'z_right': -2.250000, 'slope': 0.085267, 'intercept': 0.292058},
    {'z_left': -2.250000, 'z_right': -2.000000, 'slope': 0.106886, 'intercept': 0.340700},
    {'z_left': -2.000000, 'z_right': -1.750000, 'slope': 0.133185, 'intercept': 0.393297},
    {'z_left': -1.750000, 'z_right': -1.500000, 'slope': 0.164757, 'intercept': 0.448548},
    {'z_left': -1.500000, 'z_right': -1.250000, 'slope': 0.202063, 'intercept': 0.504508},
    {'z_left': -1.250000, 'z_right': -1.000000, 'slope': 0.245330, 'intercept': 0.558592},
    {'z_left': -1.000000, 'z_right': -0.750000, 'slope': 0.294437, 'intercept': 0.607699},
    {'z_left': -0.750000, 'z_right': -0.500000, 'slope': 0.348824, 'intercept': 0.648489},
    {'z_left': -0.500000, 'z_right': -0.250000, 'slope': 0.407450, 'intercept': 0.677802},
    {'z_left': -0.250000, 'z_right': 0.000000, 'slope': 0.468831, 'intercept': 0.693147}
]

def log_add_exp_piecewise_approx_niti(
    a_scaled: int, 
    b_scaled: int, 
    scale_a: float, 
    scale_b: float
) -> Tuple[int, float]:
    """
    Compute log(exp(a) + exp(b)) using piecewise linear approximation for NITI.
    
    Args:
        a_scaled: int8 representation of log value a (scaled by scale_a)
        b_scaled: int8 representation of log value b (scaled by scale_b)
        scale_a: actual scale factor for a (power of two from block scaling)
        scale_b: actual scale factor for b (power of two from block scaling)
        
    Returns:
        Tuple of (result_scaled, output_scale) where:
        - result_scaled: int8 approximation of log(exp(a) + exp(b)) scaled by output_scale
        - output_scale: scale factor to apply to result_scaled to get actual log value
    """
    # Convert scaled integers to actual log values
    # Note: In real NITI, scale factors are powers of two, so this can be bit shifts
    a_log = a_scaled * scale_a
    b_log = b_scaled * scale_b
    
    # Ensure a_log >= b_log for numerical stability
    if a_log < b_log:
        a_log, b_log = b_log, a_log
        scale_a, scale_b = scale_b, scale_a  # Swap scales accordingly
    
    # Compute z = b_log - a_log (so z <= 0)
    z = b_log - a_log
    
    # Clamp z to approximation domain [-10, 0]
    Z_MIN = -10.0
    Z_MAX = 0.0
    if z < Z_MIN:
        z = Z_MIN
    elif z > Z_MAX:
        z = Z_MAX
    
    # Find the appropriate segment
    NUM_SEGMENTS = 40
    seg_idx = int((z - Z_MIN) / (Z_MAX - Z_MIN) * NUM_SEGMENTS)
    seg_idx = max(0, min(seg_idx, NUM_SEGMENTS - 1))
    
    seg = _SEGMENTS_NITI[seg_idx]
    
    # Approximate log(1 + exp(z)) = slope * z + intercept
    log1p_exp_z_approx = seg['slope'] * z + seg['intercept']
    
    # Result in log domain: a_log + log(1 + exp(b-a))
    result_log = a_log + log1p_exp_z_approx
    
    # Determine output scale - for simplicity, we'll use the larger of the two input scales
    # In practice, this might be determined by the layer or operation context
    output_scale = max(scale_a, scale_b)
    
    # Convert back to scaled integer (int8 range)
    # Clamp to int8 range [-128, 127]
    result_scaled = round(result_log / output_scale)
    result_scaled = max(-128, min(127, result_scaled))
    
    return result_scaled, output_scale

def log_sum_exp_piecewise_approx_niti(log_values_scaled: List[int], scale: float) -> Tuple[int, float]:
    """
    Compute log-sum-exp for a list of scaled log values using the approximation sequentially.
    
    Args:
        log_values_scaled: List of int8 log values (all scaled by the same scale factor)
        scale: common scale factor for all values
        
    Returns:
        Tuple of (result_scaled, output_scale)
    """
    if not log_values_scaled:
        return -128, scale  # Representing -inf
    
    acc_scaled = log_values_scaled[0]
    acc_scale = scale
    
    for i in range(1, len(log_values_scaled)):
        acc_scaled, acc_scale = log_add_exp_piecewise_approx_niti(
            acc_scaled, log_values_scaled[i], acc_scale, scale
        )
    
    return acc_scaled, acc_scale

if __name__ == "__main__":
    # Example usage
    print("NITI piecewise linear log-sum-exp approximation")
    print("Example: log(exp(0.0) + exp(-1.0))")
    
    # Using Q4.4 format (scale = 1/16)
    scale = 1.0 / 16.0
    a_scaled = round(0.0 / scale)  # 0.0
    b_scaled = round(-1.0 / scale)  # -1.0
    
    result_scaled, result_scale = log_add_exp_piecewise_approx_niti(a_scaled, b_scaled, scale, scale)
    result_log = result_scaled * result_scale
    
    import math
    exact = math.log(math.exp(0.0) + math.exp(-1.0))
    
    print(f"a_scaled: {a_scaled}, b_scaled: {b_scaled}")
    print(f"Result scaled: {result_scaled}, Result scale: {result_scale}")
    print(f"Approximated log-sum-exp: {result_log}")
    print(f"Exact log-sum-exp: {exact}")
    print(f"Error: {abs(result_log - exact)}")
