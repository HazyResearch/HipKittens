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
# Benchmarking
# **************************************************

num_warmup = 20
num_iters = 20
start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)

def flops(batch, seqlen, nheads, headdim, causal, mode="bwd"):
    assert mode in ["fwd", "bwd", "fwd_bwd"]
    f = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return f if mode == "fwd" else (2.5 * f if mode == "bwd" else 3.5 * f)

def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time


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

    return dQ, dK, dV, Delta

# **************************************************
# Generate inputs
# **************************************************


causal = False
b = 1
h = 1
n = 32
d = 128
dtype = torch.bfloat16
mean = 10
std = 0.1  

flops_ref = flops(b, n, h, d, causal, mode="bwd")

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
# AITER forward and backward
# **************************************************

if use_aiter:
    timings = []
    print("\nRunning AITER...")

    for _ in range(num_warmup):
        Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        dO_aiter = dO_bhnd.transpose(1, 2).contiguous()
        out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal, return_lse=True, deterministic=True)
        out_aiter.backward(dO_aiter)
    
    for _ in range(num_iters):
        Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        dO_aiter = dO_bhnd.transpose(1, 2).contiguous()
        out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal, return_lse=True, deterministic=True)
        torch.cuda.synchronize()
        start_event.record()
        out_aiter.backward(dO_aiter)
        end_event.record()
        torch.cuda.synchronize()
        elapsed_time = start_event.elapsed_time(end_event)
        timings.append(elapsed_time)

    avg_time_aiter = sum(timings) / len(timings)
    eff_aiter = efficiency(flops_ref, avg_time_aiter)
    print(f"AITER (AMD) reference average execution time: {avg_time_aiter:.4f} ms")
    print(f"AITER (AMD) reference performance: {eff_aiter:.2f} TFLOPS for {b=} {h=} {n=} {d=} {causal=}.\n")

    q_grad_aiter_bnhd = Q_aiter.grad
    k_grad_aiter_bnhd = K_aiter.grad  
    v_grad_aiter_bnhd = V_aiter.grad
    out_aiter_bhnd = out_aiter.transpose(1, 2)  # BNHD -> BHND
    q_grad_aiter_bhnd = q_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
    k_grad_aiter_bhnd = k_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
    v_grad_aiter_bhnd = v_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND

# **************************************************
# PyTorch Reference
# **************************************************

print("Running PyTorch reference...")
timings = []
for _ in range(num_warmup):
    Q_pytorch = Q_bhnd.clone().detach().requires_grad_(True)
    K_pytorch = K_bhnd.clone().detach().requires_grad_(True)
    V_pytorch = V_bhnd.clone().detach().requires_grad_(True)
    dO_pytorch = dO_bhnd.clone()
    out_pytorch, q_pytorch, k_pytorch, v_pytorch = reference_forward(Q_pytorch, K_pytorch, V_pytorch, causal)
    out_pytorch.backward(dO_pytorch)

for _ in range(num_iters):
    Q_pytorch = Q_bhnd.clone().detach().requires_grad_(True)
    K_pytorch = K_bhnd.clone().detach().requires_grad_(True)
    V_pytorch = V_bhnd.clone().detach().requires_grad_(True)
    dO_pytorch = dO_bhnd.clone()
    out_pytorch, q_pytorch, k_pytorch, v_pytorch = reference_forward(Q_pytorch, K_pytorch, V_pytorch, causal)
    torch.cuda.synchronize()
    start_event.record()
    out_pytorch.backward(dO_pytorch)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)

avg_time_pytorch = sum(timings) / len(timings)
eff_pytorch = efficiency(flops_ref, avg_time_pytorch)
print(f"PyTorch reference average execution time: {avg_time_pytorch:.4f} ms")
print(f"PyTorch reference performance: {eff_pytorch:.2f} TFLOPS for {b=} {h=} {n=} {d=} {causal=}.\n")

q_grad_pytorch = q_pytorch.grad
k_grad_pytorch = k_pytorch.grad
v_grad_pytorch = v_pytorch.grad

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

dQ_tiled, dK_tiled, dV_tiled, delta_tiled = simple_flash_backward(Q_tiled.float(), K_tiled.float(), V_tiled.float(), dO_tiled.float(), m_tiled, l_tiled)
out_tiled_bhnd = O_tiled
q_grad_tiled_bhnd = dQ_tiled
k_grad_tiled_bhnd = dK_tiled
v_grad_tiled_bhnd = dV_tiled


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
    P = torch.exp(S - m.unsqueeze(-1)) / l.unsqueeze(-1)
    O = torch.matmul(P, V)
    # # softmax backward
    Delta = (dO * O).sum(dim=-1, keepdim=True)                 #
    dS = P * (torch.matmul(dO, V.transpose(-2, -1)) - Delta)   # (B, N, H, N)
    # chain rule through S = (Q K^T) * scale
    dQ = torch.matmul(dS, K) * scale
    return dQ

def test_dv(Q, K, V, dO, m, l):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.exp(S - m.unsqueeze(-1)) / l.unsqueeze(-1)
    O = torch.matmul(P, V)
    # dV
    dV = torch.matmul(P.transpose(-2, -1), dO)
    return dV


def test_dk(Q, K, V, dO, m, l):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K.transpose(-2, -1)) * scale
    P = torch.exp(S- m.unsqueeze(-1)) / l.unsqueeze(-1)
    O = torch.matmul(P, V)
    # softmax backward
    Delta = (dO * O).sum(dim=-1, keepdim=True)                 # (B, N, H, 1)
    dS = P * (torch.matmul(dO, V.transpose(-2, -1)) - Delta)   # (B, N, H, N)
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale
    return dK

DQ = test_dq(Q_tiled.float(), K_tiled.float(), V_tiled.float(), dO_tiled.float(), m_tiled, l_tiled)
DV = test_dv(Q_tiled.float(), K_tiled.float(), V_tiled.float(), dO_tiled.float(), m_tiled, l_tiled)
DK = test_dk(Q_tiled.float(), K_tiled.float(), V_tiled.float(), dO_tiled.float(), m_tiled, l_tiled)

# TK
print("Running ThunderKittens ...")
timings = []
for _ in range(num_warmup):
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
        dO_tk,    
        dQ_tk,   
        dK_tk,    
        dV_tk,    
        m_tk, 
        l_tk,
        delta_tk
    )


for _ in range(num_iters):
    dQ_tk = torch.zeros_like(q_grad_tiled_bhnd).float()
    dK_tk = torch.zeros_like(k_grad_tiled_bhnd).float()
    dV_tk = torch.zeros_like(v_grad_tiled_bhnd).float()
    torch.cuda.synchronize()
    start_event.record()

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
        dO_tk,    
        dQ_tk,   
        dK_tk,    
        dV_tk,    
        m_tk, 
        l_tk,
        delta_tk
    )

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)

avg_time_tk = sum(timings) / len(timings)
eff_tk = efficiency(flops_ref, avg_time_tk)
print(f"ThunderKittens average execution time: {avg_time_tk:.4f} ms")
print(f"ThunderKittens performance: {eff_tk:.2f} TFLOPS for {b=} {h=} {n=} {d=} {causal=}.\n")

# **************************************************
# Comparisons
# **************************************************

if use_aiter:
    out_diff = (out_aiter_bhnd - out_pytorch).abs()
    q_grad_diff = (q_grad_aiter_bhnd - q_grad_pytorch).abs()
    k_grad_diff = (k_grad_aiter_bhnd - k_grad_pytorch).abs()
    v_grad_diff = (v_grad_aiter_bhnd - v_grad_pytorch).abs()

# Compare TK with PyTorch
out_tiled_diff = (out_tiled_bhnd - out_pytorch).abs()
q_grad_tiled_diff = (q_grad_tiled_bhnd - q_grad_pytorch).abs()
k_grad_tiled_diff = (k_grad_tiled_bhnd - k_grad_pytorch).abs()
v_grad_tiled_diff = (v_grad_tiled_bhnd - v_grad_pytorch).abs()

if use_aiter:
    print(f"\nAITER vs PyTorch comparison:")
    print(f"Output max error: {out_diff.max().item():.6f}")
    print(f"Q grad max error: {q_grad_diff.max().item():.6f}")
    print(f"K grad max error: {k_grad_diff.max().item():.6f}")
    print(f"V grad max error: {v_grad_diff.max().item():.6f}")

print(f"\nTiled vs PyTorch comparison:")
print(f"Output max error: {out_tiled_diff.max().item():.6f}")
print(f"Q grad max error: {q_grad_tiled_diff.max().item():.6f}")
print(f"K grad max error: {k_grad_tiled_diff.max().item():.6f}")
print(f"V grad max error: {v_grad_tiled_diff.max().item():.6f}")

# TK vs PyTorch
print(f"\nTK vs PyTorch comparison:")

num_print = 8
print("\nGradient K outputs:")
print("TK: ", dK_tk[0, 0, 0, :num_print], "Max:", dK_tk.max().item())
print("PyTorch: ", DK[0, 0, 0, :num_print], "Max:", DK.max().item())
if use_aiter:
    print("AITER: ", k_grad_aiter_bnhd[0, 0, 0, :num_print], "Max:", k_grad_aiter_bnhd.max().item())

print()
print("Gradient V outputs:")
print("TK: ", dV_tk[0, 0, 0, :num_print], "Max:", dV_tk.max().item())
print("PyTorch: ", DV[0, 0, 0, :num_print], "Max:", DV.max().item())
if use_aiter:
    print("AITER: ", v_grad_aiter_bnhd[0, 0, 0, :num_print], "Max:", v_grad_aiter_bnhd.max().item())

print()
print("Gradient Q outputs:")
print("TK: ", dQ_tk[0, 0, 0, :num_print], "Max:", dQ_tk.max().item())
print("PyTorch: ", DQ[0, 0, 0, :num_print], "Max:", DQ.max().item())
if use_aiter:
    print("AITER: ", q_grad_aiter_bnhd[0, 0, 0, :num_print], "Max:", q_grad_aiter_bnhd.max().item())


# **************************************************
# TK vs PyTorch (robust tolerances & metrics)
# **************************************************
print(f"\nRobustness checks (TK vs PyTorch):")

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
    return diff, error_count, numel, rel_error, l2_error, cos   

# Compute diffs in float32 to avoid bf16 quantization in the comparison itself
delta_diff, delta_err_cnt, delta_total, delta_rel_error, delta_l2_error, delta_cos = robustness_check(delta_tiled, delta_tk)
q_diff, q_err_cnt, q_total, q_rel_error, q_l2_error, q_cos = robustness_check(DQ, dQ_tk)
k_diff, k_err_cnt, k_total, k_rel_error, k_l2_error, k_cos = robustness_check(DK, dK_tk)
v_diff, v_err_cnt, v_total, v_rel_error, v_l2_error, v_cos = robustness_check(DV, dV_tk)

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


