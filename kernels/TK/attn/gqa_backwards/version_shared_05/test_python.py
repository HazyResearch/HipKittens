import torch
import random
import math
import tk_kernel
import time

use_aiter = True
if use_aiter:
    import aiter

torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(
    precision=3,        
    sci_mode=False,     
    linewidth=220,      
    threshold=float("inf")  
)

# **************************************************
# Reference
# **************************************************

def reference_forward(Q, K, V, causal):
    """Reference implementation using BHND layout (batch, heads, seq, dim)"""
    # Convert to float64 and create new leaf tensors with requires_grad
    q_ = Q.detach().to(torch.float64).requires_grad_(True)
    k_ = K.detach().to(torch.float64).requires_grad_(True)
    v_ = V.detach().to(torch.float64).requires_grad_(True)
    
    # manual pytorch implementation of scaled dot product attention
    QK = torch.matmul(q_, k_.transpose(-2, -1))
    QK /= (q_.size(-1) ** 0.5)
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_)
    
    return output, q_, k_, v_

def simple_flash_backward(Q, K, V, dO, m, l):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)

    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.exp(S - m.unsqueeze(-1)) / l.unsqueeze(-1)
    O = torch.matmul(P, V)

    # dV
    dV = torch.matmul(P.transpose(-2, -1), dO)

    # softmax backward
    Delta = (dO * O).sum(dim=-1, keepdim=True)                 # (B, N, H, 1)
    dS = P * (torch.matmul(dO, V.transpose(-2, -1)) - Delta)   # (B, N, H, N)

    # chain rule through S = (Q K^T) * scale
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale

    return P, dQ, dK, dV, Delta

# **************************************************
# Generate inputs
# **************************************************


causal = False
b = 1
h = 1
n = 1024
d = 128
dtype = torch.bfloat16
mean = 10
std = 0.1  

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()

def generate_inputs():
    # Generate in BHND format (batch, heads, seq, dim) for reference
    Q = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    K = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    V = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda')
    dO = generate_tensor((b, h, n, d), mean, std, torch.bfloat16, 'cuda') 

    Q.requires_grad_(True)
    K.requires_grad_(True)
    V.requires_grad_(True)
    return Q, K, V, dO

# Generate base inputs in BHND format
Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = generate_inputs()

# **************************************************
# Tiled Reference
# **************************************************

print("Running Tiled forward to get m, l...\n")
Q_tiled = Q_bhnd.clone().contiguous().detach().requires_grad_(True)  
K_tiled = K_bhnd.clone().contiguous().detach().requires_grad_(True)  
V_tiled = V_bhnd.clone().contiguous().detach().requires_grad_(True)  
dO_tiled = dO_bhnd.clone().contiguous()  
QK = torch.matmul(Q_tiled.float(), K_tiled.transpose(-2, -1).float()) / math.sqrt(d)
m_tiled = QK.max(dim=-1, keepdim=True)[0] 
exp_scores = torch.exp(QK - m_tiled)  
l_tiled = exp_scores.sum(dim=-1, keepdim=True)  
P_tiled = exp_scores / l_tiled
O_tiled = torch.matmul(P_tiled, V_tiled.float())
m_tiled = m_tiled.squeeze(-1)
l_tiled = l_tiled.squeeze(-1)

P_tiled, dQ_tiled, dK_tiled, dV_tiled, delta_tiled = simple_flash_backward(Q_tiled.float(), K_tiled.float(), V_tiled.float(), dO_tiled.float(), m_tiled, l_tiled)
out_tiled_bhnd = O_tiled
q_grad_tiled_bhnd = dQ_tiled
k_grad_tiled_bhnd = dK_tiled
v_grad_tiled_bhnd = dV_tiled
delta_tiled = delta_tiled.transpose(-1, -2).contiguous()


# **************************************************
# ThunderKittens
# **************************************************

# Get forwards pass outputs
Q_tk = Q_bhnd.bfloat16().clone().contiguous().detach().requires_grad_(True)  
K_tk = K_bhnd.bfloat16().clone().contiguous().detach().requires_grad_(True)  
V_tk = V_bhnd.bfloat16().clone().contiguous().detach().requires_grad_(True)  
O_tk = O_tiled.bfloat16().clone()
dO_tk = dO_bhnd.float().clone()
m_tk = m_tiled.float().unsqueeze(-1)
l_tk = l_tiled.float().unsqueeze(-1)

def test_dq(Q, K, V, dO, m, l):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.exp(S - m) / l
    O = torch.matmul(P.float(), V.float())
    # # softmax backward
    Delta = (dO * O).sum(dim=-1, keepdim=True)             
    dS = (torch.matmul(dO.float(), V.transpose(-2, -1).float()) - Delta) * P.float()  * scale  # (B, N, H, N)
    # chain rule through S = (Q K^T) * scale
    dQ = torch.matmul(torch.ones_like(dS.float()), K.float())
    dV = torch.matmul(P.float().transpose(-2, -1), dO)
    return P

dS_test = test_dq(Q_tk.clone(), K_tk.clone(), V_tk.clone(), dO_tk.clone(), m_tk.clone(), l_tk.clone())
dS_ij_tk = torch.zeros_like(dS_test.bfloat16().clone())

# TK
print("Running ThunderKittens ...")
dQ_tk = torch.zeros_like(q_grad_tiled_bhnd).float()
dK_tk = torch.zeros_like(k_grad_tiled_bhnd).float()
dV_tk = torch.zeros_like(v_grad_tiled_bhnd).float()
delta_tk = torch.zeros_like(delta_tiled).float()

tk_kernel.dispatch_prep(
    O_tk,     # Og
    dO_tk,    # dOg
    delta_tk, # delta
)

tk_kernel.dispatch_bwd_combined(
    Q_tk,     
    K_tk,     
    V_tk,     
    O_tk,   
    dS_ij_tk,
    dO_tk,    
    dQ_tk,   
    dK_tk,    
    dV_tk,    
    m_tk, 
    l_tk,
    delta_tk
)

# **************************************************
# Comparisons
# **************************************************

# TK vs Tiled
print(f"\nTK vs Tiled comparison:")

num_print = 17
print("\nDelta outputs:")
print(f"TK: {delta_tk[0, 0, 0, :num_print]}")
print(f"Delta: {delta_tiled[0, 0, 0, :num_print]}")

print("\nGradient K outputs:")
print("TK: ", dK_tk[0, 0, 0, :num_print], "Max:", dK_tk.max().item())
print("Tiled: ", k_grad_tiled_bhnd[0, 0, 0, :num_print], "Max:", k_grad_tiled_bhnd.max().item())

print()
print("Gradient V outputs:")
print("TK: ", dV_tk[0, 0, 0, :num_print], "Max:", dV_tk.max().item())
print("Tiled: ", v_grad_tiled_bhnd[0, 0, 0, :num_print], "Max:", v_grad_tiled_bhnd.max().item())

print()
print("Gradient Q outputs:")
print("TK: ", dQ_tk[0, 0, 0, :num_print], "Max:", dQ_tk.max().item())
print("Tiled: ", q_grad_tiled_bhnd[0, 0, 0, :num_print], "Max:", q_grad_tiled_bhnd.max().item())

# **************************************************
# TK vs Tiled (robust tolerances & metrics)
# **************************************************
print(f"\nRobustness checks (TK vs Tiled):")

def robustness_check(ref, pred):
    ref = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask = (diff > (0.001 + 0.05 * denom))
    error_count = mask.sum().item()
    numel = ref.numel()
    rel_error = error_count / numel
    l2_error = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), pred.flatten(), dim=0).item()
    return diff, error_count, numel, rel_error, l2_error, cos, mask   

# Compute diffs in float32 to avoid bf16 quantization in the comparison itself
delta_diff, delta_err_cnt, delta_total, delta_rel_error, delta_l2_error, delta_cos, delta_mask = robustness_check(delta_tiled, delta_tk)
q_diff, q_err_cnt, q_total, q_rel_error, q_l2_error, q_cos, q_mask = robustness_check(q_grad_tiled_bhnd, dQ_tk)
k_diff, k_err_cnt, k_total, k_rel_error, k_l2_error, k_cos, k_mask = robustness_check(k_grad_tiled_bhnd, dK_tk)
v_diff, v_err_cnt, v_total, v_rel_error, v_l2_error, v_cos, v_mask = robustness_check(v_grad_tiled_bhnd, dV_tk)

print(f"Delta: max_abs={delta_diff.max().item():.6f}, max_rel={delta_rel_error:.4f}, "
      f"rel_l2={delta_l2_error:.4f}, cos={delta_cos:.6f}, "
      f"errors={delta_err_cnt}/{delta_total} ({100*delta_err_cnt/delta_total:.4f}%)")
print(f"Q grad: max_abs={q_diff.max().item():.6f}, max_rel={q_rel_error:.4f}, "
        f"rel_l2={q_l2_error:.4f}, cos={q_cos:.6f}, "
      f"errors={q_err_cnt}/{q_total} ({100*q_err_cnt/q_total:.4f}%)")
print(f"K grad: max_abs={k_diff.max().item():.6f}, max_rel={k_rel_error:.4f}, "
      f"rel_l2={k_l2_error:.4f}, cos={k_cos:.6f}, "
      f"errors={k_err_cnt}/{k_total} ({100*k_err_cnt/k_total:.4f}%)")
print(f"V grad: max_abs={v_diff.max().item():.6f}, max_rel={v_rel_error:.4f}, "
      f"rel_l2={v_l2_error:.4f}, cos={v_cos:.6f}, "
      f"errors={v_err_cnt}/{v_total} ({100*v_err_cnt/v_total:.4f}%)")


