import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SHARQ_OPS = None

def quantize_s1p2(tensor):

    representable_vals = torch.tensor([
        -1.75, -1.5, -1.25, -1.0, -0.75, -0.5, -0.25, 0.0,
        0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75
    ], device=tensor.device, dtype=tensor.dtype)
    
    diff = torch.abs(tensor.unsqueeze(-1) - representable_vals)
    indices = torch.argmin(diff, dim=-1)
    return representable_vals[indices]

def quantize_e6m2(tensor):

    tensor = torch.clamp(tensor, min=2.0**(-48), max=2.0**15 * 1.5)
    
    exponent = torch.floor(torch.log2(tensor))
    mantissa_val = tensor / (2.0**exponent) - 1.0 
    
    quantized_mantissa_val = torch.round(mantissa_val * 4.0) / 4.0
    
    overflow = quantized_mantissa_val >= 1.0
    quantized_mantissa_val[overflow] = 0.0
    exponent[overflow] += 1.0
    
    exponent = torch.clamp(exponent, min=-48, max=15)
    
    is_nan = (exponent == 15) & (quantized_mantissa_val == 0.75)
    quantized_mantissa_val[is_nan] = 0.5
    
    reconstructed_val = (1.0 + quantized_mantissa_val) * (2.0**exponent)
    return reconstructed_val

def quantize_hif4_tensor(tensor, group_size=64):
    original_shape = tensor.shape
    
    padding = (group_size - tensor.shape[-1] % group_size) % group_size
    if padding != 0:
        tensor = F.pad(tensor, (0, padding))
        
    reshaped_tensor = tensor.view(-1, group_size)
    N = reshaped_tensor.shape[0]
    
    V16 = torch.max(torch.abs(reshaped_tensor.view(N, 16, 4)), dim=2)[0]
    V8 = torch.max(V16.view(N, 8, 2), dim=2)[0]
    Vmax = torch.max(V8, dim=1, keepdim=True)[0]
    
    SF_BF16 = Vmax / 7.0
    SF_BF16[SF_BF16 == 0] = 2.0**(-48)
    del Vmax  
    
    E6M2 = quantize_e6m2(SF_BF16)
    E6M2_REC = 1.0 / E6M2
    del SF_BF16 
    
    E1_8 = (V8 * E6M2_REC >= 4.0).float()
    del V8 
    
    E1_8_expanded = E1_8.repeat_interleave(2, dim=1) 
    E1_16 = (V16 * E6M2_REC * (2.0 ** (-E1_8_expanded)) >= 2.0).float()
    del V16, E1_8_expanded 
    
    E1_8_full = E1_8.repeat_interleave(8, dim=1)
    E1_16_full = E1_16.repeat_interleave(4, dim=1)
    
    V64_scaled = reshaped_tensor * E6M2_REC * (2.0 ** (-E1_8_full)) * (2.0 ** (-E1_16_full))
    
    S1P2_64 = quantize_s1p2(V64_scaled)
    del V64_scaled 
    
    dequantized_tensor_groups = S1P2_64 * E6M2 * (2.0 ** E1_8_full) * (2.0 ** E1_16_full)
    del S1P2_64, E1_8_full, E1_16_full, E6M2, E6M2_REC 
    
    dequantized_tensor = dequantized_tensor_groups.view(tensor.shape)
    if padding != 0:
        dequantized_tensor = dequantized_tensor[..., :-padding]
        
    return dequantized_tensor.view(original_shape)