import torch
import random
import math
import tk_kernel_fwd
import tk_kernel_bkwd
import time

use_aiter = True
if use_aiter:
    import aiter

torch.cuda.set_device(7)
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

num_warmup = 50
num_iters = 100
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


# **************************************************
# Reference
# **************************************************

def expand_kv_for_gqa(K, V, h_q, h_kv):
    """Expand K,V from h_kv heads to h_q heads for GQA by replicating each KV head"""
    group_size = h_q // h_kv
    # Repeat each KV head group_size times: (B, h_kv, N, D) -> (B, h_q, N, D)
    K_expanded = K.repeat_interleave(group_size, dim=1)  
    V_expanded = V.repeat_interleave(group_size, dim=1)
    return K_expanded, V_expanded

def reference_forward(Q, K, V, causal):
    """GQA Reference implementation using BHND layout (batch, heads, seq, dim)"""
    # Convert to float64 and create new leaf tensors with requires_grad
    q_ = Q.detach().to(torch.float64).requires_grad_(True)
    k_ = K.detach().to(torch.float64).requires_grad_(True) 
    v_ = V.detach().to(torch.float64).requires_grad_(True)
    
    # Expand K,V to match Q heads for GQA computation
    k_expanded, v_expanded = expand_kv_for_gqa(k_, v_, h_q, h_kv)
    
    # Manual pytorch implementation of scaled dot product attention
    QK = torch.matmul(q_, k_expanded.transpose(-2, -1))
    QK /= (q_.size(-1) ** 0.5)
    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_expanded)
    
    return output, q_, k_, v_

def simple_flash_backward(Q, K, V, dO, L):
    """GQA version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)
    
    # Expand K,V to match Q heads for GQA computation  
    K_expanded, V_expanded = expand_kv_for_gqa(K, V, h_q, h_kv)

    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale
    P = torch.exp(S - L.unsqueeze(-1))
    O = torch.matmul(P, V_expanded)

    # dV - need to sum across grouped heads  
    dV_expanded = torch.matmul(P.transpose(-2, -1), dO)  # (B, h_q, N, D)
    dV = torch.zeros_like(V)
    group_size = h_q // h_kv
    for i in range(h_kv):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        dV[:, i, :, :] = dV_expanded[:, start_idx:end_idx, :, :].sum(dim=1)

    # softmax backward
    Delta = (dO * O).sum(dim=-1, keepdim=True)                 # (B, h_q, N, 1)
    dS = P * (torch.matmul(dO, V_expanded.transpose(-2, -1)) - Delta)   # (B, h_q, N, N)

    # chain rule through S = (Q K^T) * scale  
    dQ = torch.matmul(dS, K_expanded) * scale  # (B, h_q, N, D)
    
    # dK - need to sum across grouped heads
    dK_expanded = torch.matmul(dS.transpose(-2, -1), Q) * scale  # (B, h_q, N, D)
    dK = torch.zeros_like(K)
    for i in range(h_kv):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size 
        dK[:, i, :, :] = dK_expanded[:, start_idx:end_idx, :, :].sum(dim=1)

    return dQ, dK, dV, Delta

# **************************************************
# Generate inputs
# **************************************************


causal = False
b = 16
h_q = 64  # number of query heads  
h_kv = 8  # number of key/value heads (for GQA)
group_size = h_q // h_kv  # queries per KV head group
n = 1024
d = 128
dtype = torch.bfloat16
mean = 10
std = 0.1  

flops_ref = flops(b, n, h_q, d, causal, mode="bwd")  # Use query heads for FLOP calculation

def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()

def generate_inputs():
    # Generate in BHND format (batch, heads, seq, dim) for GQA
    # Q has h_q heads, but K and V have h_kv heads
    Q = generate_tensor((b, h_q, n, d), mean, std, torch.bfloat16, 'cuda')
    K = generate_tensor((b, h_kv, n, d), mean, std, torch.bfloat16, 'cuda') 
    V = generate_tensor((b, h_kv, n, d), mean, std, torch.bfloat16, 'cuda')
    dO = generate_tensor((b, h_q, n, d), mean, std, torch.bfloat16, 'cuda') 

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
        out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal, return_lse=True, deterministic=False)
        out_aiter.backward(dO_aiter)
    
    for _ in range(num_iters):
        Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
        dO_aiter = dO_bhnd.transpose(1, 2).contiguous()
        out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal, return_lse=True, deterministic=False)
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
    print(f"AITER (AMD) reference performance: {eff_aiter:.2f} TFLOPS for {b=} h_q={h_q} h_kv={h_kv} {n=} {d=} {causal=}.\n")

    q_grad_aiter_bnhd = Q_aiter.grad
    k_grad_aiter_bnhd = K_aiter.grad  
    v_grad_aiter_bnhd = V_aiter.grad
    out_aiter_bhnd = out_aiter
    # out_aiter_bhnd = out_aiter.transpose(1, 2)  # BNHD -> BHND
    # q_grad_aiter_bhnd = q_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
    # k_grad_aiter_bhnd = k_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
    # v_grad_aiter_bhnd = v_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND

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
print(f"PyTorch reference performance: {eff_pytorch:.2f} TFLOPS for {b=} h_q={h_q} h_kv={h_kv} {n=} {d=} {causal=}.\n")

q_grad_pytorch = q_pytorch.grad
k_grad_pytorch = k_pytorch.grad
v_grad_pytorch = v_pytorch.grad
out_pytorch = out_pytorch.transpose(1, 2) # BHND -> BNHD
q_grad_pytorch = q_grad_pytorch.transpose(1, 2) # BHND -> BNHD
k_grad_pytorch = k_grad_pytorch.transpose(1, 2) # BHND -> BNHD
v_grad_pytorch = v_grad_pytorch.transpose(1, 2) # BHND -> BNHD

# **************************************************
# Tiled Reference
# **************************************************

print("Running Tiled forward to get L...\n")
Q_tiled = Q_bhnd.clone().contiguous().detach().requires_grad_(True)  
K_tiled = K_bhnd.clone().contiguous().detach().requires_grad_(True)  
V_tiled = V_bhnd.clone().contiguous().detach().requires_grad_(True)  
dO_tiled = dO_bhnd.clone().contiguous()  

# Expand K,V for GQA computation
K_tiled_expanded, V_tiled_expanded = expand_kv_for_gqa(K_tiled.float(), V_tiled.float(), h_q, h_kv)

QK = torch.matmul(Q_tiled.float(), K_tiled_expanded.transpose(-2, -1)) / math.sqrt(d)
m_tiled = QK.max(dim=-1, keepdim=True)[0] 
exp_scores = torch.exp(QK - m_tiled)  
l_tiled = exp_scores.sum(dim=-1, keepdim=True)  
P_tiled = exp_scores / l_tiled
O_tiled = torch.matmul(P_tiled, V_tiled_expanded)
L_tiled = (m_tiled + torch.log(l_tiled)).squeeze(-1)
# L_tiled = torch.log(l_tiled).squeeze(-1)
# L_tiled = m_tiled.squeeze(-1)

dQ_tiled, dK_tiled, dV_tiled, delta_tiled = simple_flash_backward(Q_tiled.float(), K_tiled.float(), V_tiled.float(), dO_tiled.float(), L_tiled)
out_tiled_bnhd = O_tiled.transpose(1, 2) # BHND -> BNHD
q_grad_tiled_bnhd = dQ_tiled.transpose(1, 2) # BHND -> BNHD
k_grad_tiled_bnhd = dK_tiled.transpose(1, 2) # BHND -> BNHD
v_grad_tiled_bnhd = dV_tiled.transpose(1, 2) # BHND -> BNHD
L_tiled = L_tiled.unsqueeze(-1)

# **************************************************
# ThunderKittens
# **************************************************

# Get forwards pass outputs
Q_tk = Q_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True)  
K_tk = K_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True)  
V_tk = V_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True) 
dO_tk = dO_bhnd.transpose(1, 2).bfloat16().clone().contiguous()

# Call TK forward to get O and L
O_tk = torch.zeros_like(out_tiled_bnhd).bfloat16().clone().contiguous()
L_tk = torch.zeros_like(L_tiled).float().transpose(-1, -2).contiguous()

tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tk, L_tk)
torch.cuda.synchronize()

# L_tk = L_tiled.float().contiguous()

# TK
print("Running ThunderKittens ...")
timings = []
for _ in range(num_warmup):
    dQ_tk_in = torch.zeros_like(q_grad_tiled_bnhd).bfloat16().transpose(1, 2).contiguous()
    dQ_tk = torch.zeros_like(q_grad_tiled_bnhd).bfloat16().contiguous()
    dK_tk = torch.zeros_like(k_grad_tiled_bnhd).bfloat16().contiguous()
    dV_tk = torch.zeros_like(v_grad_tiled_bnhd).bfloat16().contiguous()
    delta_tk = torch.zeros_like(delta_tiled).float().transpose(-1, -2).contiguous()

    tk_kernel_bkwd.dispatch_prep(
        O_tk,     # Og
        dO_tk,    # dOg
        delta_tk, # delta
    )

    tk_kernel_bkwd.dispatch_bwd_combined(
        Q_tk,     
        K_tk,     
        V_tk,     
        O_tk,     
        dO_tk,    
        dQ_tk_in,   
        dK_tk,    
        dV_tk,    
        L_tk,
        delta_tk
    )

    tk_kernel_bkwd.dispatch_dq_shuffle(
        dQ_tk_in,
        dQ_tk
    )


for _ in range(num_iters):
    dQ_tk_in = torch.empty(q_grad_tiled_bnhd.shape).bfloat16().transpose(1, 2).contiguous().to('cuda')
    dQ_tk = torch.empty(q_grad_tiled_bnhd.shape).bfloat16().contiguous().to('cuda')
    dK_tk = torch.empty(k_grad_tiled_bnhd.shape).bfloat16().contiguous().to('cuda')
    dV_tk = torch.empty(v_grad_tiled_bnhd.shape).bfloat16().contiguous().to('cuda')
    # delta_tk = torch.zeros_like(delta_tiled).float()
    delta_tk = torch.empty(delta_tiled.shape).float().transpose(-1, -2).contiguous().to('cuda')
    torch.cuda.synchronize()
    start_event.record()

    tk_kernel_bkwd.dispatch_prep(
        O_tk,     # Og
        dO_tk,    # dOg
        delta_tk, # delta
    )

    tk_kernel_bkwd.dispatch_bwd_combined(
        Q_tk,     
        K_tk,     
        V_tk,     
        O_tk,     
        dO_tk,    
        dQ_tk_in,   
        dK_tk,    
        dV_tk,    
        L_tk,
        delta_tk
    )

    tk_kernel_bkwd.dispatch_dq_shuffle(
        dQ_tk_in,
        dQ_tk
    )

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
    assert not delta_tk.isnan().any()
    assert not dQ_tk_in.isnan().any()
    assert not dQ_tk.isnan().any()
    assert not dK_tk.isnan().any()
    assert not dV_tk.isnan().any()
    delta_tk = delta_tk.transpose(-1, -2).contiguous()


L_tk = L_tk.transpose(-1, -2).contiguous()

avg_time_tk = sum(timings) / len(timings)
eff_tk = efficiency(flops_ref, avg_time_tk)
print(f"ThunderKittens average execution time: {avg_time_tk:.4f} ms")
print(f"ThunderKittens performance: {eff_tk:.2f} TFLOPS for {b=} h_q={h_q} h_kv={h_kv} {n=} {d=} {causal=}.\n")

# **************************************************
# Comparisons
# **************************************************

if use_aiter:
    out_diff = (out_aiter_bhnd - out_pytorch).abs()
    q_grad_diff = (q_grad_aiter_bnhd - q_grad_pytorch).abs()
    k_grad_diff = (k_grad_aiter_bnhd - k_grad_pytorch).abs()
    v_grad_diff = (v_grad_aiter_bnhd - v_grad_pytorch).abs()

# Compare TK with PyTorch
out_tiled_diff = (out_tiled_bnhd - out_pytorch).abs()
q_grad_tiled_diff = (q_grad_tiled_bnhd - q_grad_pytorch).abs()
k_grad_tiled_diff = (k_grad_tiled_bnhd - k_grad_pytorch).abs()
v_grad_tiled_diff = (v_grad_tiled_bnhd - v_grad_pytorch).abs()

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

num_print = 8
# TK vs Tiled
print(f"\nTK vs Tiled comparison:")
print("\nO outputs:")
print("TK: ", O_tk[0, 0, :num_print, 0], "Max:", O_tk.max().item())
print("Tiled: ", out_tiled_bnhd[0, 0, :num_print, 0], "Max:", out_tiled_bnhd.max().item())
print("\nL outputs:")
print("TK: ", L_tk[0, 0, :num_print, 0], "Max:", L_tk.max().item())
print("Tiled: ", L_tiled[0, 0, :num_print, 0], "Max:", L_tiled.max().item())

# TK vs PyTorch
print(f"\nTK vs PyTorch comparison:")
print("\nDelta outputs:")
print("TK: ", delta_tk[0, 0, :num_print, 0], "Max:", delta_tk.max().item())
print("PyTorch: ", delta_tiled[0, 0, :num_print, 0], "Max:", delta_tiled.max().item())

print("\nGradient K outputs:")
print("TK: ", dK_tk[0, 0, :num_print, :num_print], "Max:", dK_tk.max().item())
print("PyTorch: ", k_grad_pytorch[0, 0, :num_print, :num_print], "Max:", k_grad_pytorch.max().item())

print()
print("Gradient V outputs:")
print("TK: ", dV_tk[0, 0, :num_print, :num_print], "Max:", dV_tk.max().item())
print("PyTorch: ", v_grad_pytorch[0, 0, :num_print, :num_print], "Max:", v_grad_pytorch.max().item())

print()
print("Gradient Q outputs:")
print("TK: ", dQ_tk[0, 0, :num_print, :num_print], "Max:", dQ_tk.max().item())
print("PyTorch: ", q_grad_pytorch[0, 0, :num_print, :num_print], "Max:", q_grad_pytorch.max().item())


# **************************************************
# TK vs Tiled (robust tolerances & metrics)
# **************************************************
# Compare O and L with tiled
print(f"\nRobustness checks (TK vs Tiled):") 
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(O_tk, out_tiled_bnhd)
l_diff, l_err_cnt, l_total, l_rel_error, l_l2_error, l_cos, l_mask = robustness_check(L_tk, L_tiled)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
print(f"L: max_abs={l_diff.max().item():.6f}, max_rel={l_rel_error:.4f}, "
      f"rel_l2={l_l2_error:.4f}, cos={l_cos:.6f}, "
      f"errors={l_err_cnt}/{l_total} ({100*l_err_cnt/l_total:.4f}%)")

# **************************************************
# TK vs PyTorch (robust tolerances & metrics)
# **************************************************
print(f"\nRobustness checks (TK vs PyTorch):") 

# Compute diffs in float32 to avoid bf16 quantization in the comparison itself
delta_diff, delta_err_cnt, delta_total, delta_rel_error, delta_l2_error, delta_cos, delta_mask = robustness_check(delta_tiled, delta_tk)
q_diff, q_err_cnt, q_total, q_rel_error, q_l2_error, q_cos, q_mask = robustness_check(q_grad_pytorch, dQ_tk)
k_diff, k_err_cnt, k_total, k_rel_error, k_l2_error, k_cos, k_mask = robustness_check(k_grad_pytorch, dK_tk)
v_diff, v_err_cnt, v_total, v_rel_error, v_l2_error, v_cos, v_mask = robustness_check(v_grad_pytorch, dV_tk)

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