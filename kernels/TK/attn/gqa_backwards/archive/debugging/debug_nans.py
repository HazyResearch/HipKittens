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
num_iters = 1

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

def reference_forward_tiled(Q, K, V, causal):
    """Tiled reference implementation with proper online softmax using BHND layout"""
    q_ = Q.detach().to(torch.bfloat16).requires_grad_(True)
    k_ = K.detach().to(torch.bfloat16).requires_grad_(True)
    v_ = V.detach().to(torch.bfloat16).requires_grad_(True)

    q_batch, q_head, q_seq, q_dim = q_.shape
    k_batch, k_head, k_seq, k_dim = k_.shape
    v_batch, v_head, v_seq, v_dim = v_.shape

    assert q_batch == k_batch == v_batch
    assert q_head == k_head == v_head
    assert q_seq == k_seq == v_seq
    assert q_dim == k_dim == v_dim

    output = torch.zeros_like(q_)
    Q_BLOCK_SIZE = 32
    KV_BLOCK_SIZE = 64
    scale = 1.0 / math.sqrt(q_dim) * 1.44269504089

    for b in range(0, q_batch):
        for h in range(0, q_head):
            for s_block in range(0, q_seq, Q_BLOCK_SIZE):
                q_block = q_[b, h, s_block:s_block+Q_BLOCK_SIZE, :]

                # Online softmax state
                max_vec = torch.full((q_block.shape[0], 1), float('-inf'), device=q_block.device, dtype=torch.float32)
                max_vec_prev = torch.full((q_block.shape[0], 1), float('-inf'), device=q_block.device, dtype=torch.float32)
                norm_vec = torch.zeros((q_block.shape[0], 1), device=q_block.device, dtype=torch.float32)
                o_reg = torch.zeros_like(q_block, dtype=torch.float32)

                for k_start in range(0, k_seq, KV_BLOCK_SIZE):
                    k_block = k_[b, h, k_start:k_start+KV_BLOCK_SIZE, :]
                    v_block = v_[b, h, k_start:k_start+KV_BLOCK_SIZE, :]

                    # Compute QK^T scaled
                    QK = torch.matmul(q_block.float(), k_block.float().transpose(-2, -1)) * scale

                    # Softmax
                    max_vec_prev = max_vec
                    max_vec = torch.max(QK, dim=-1, keepdim=True)[0]
                    QK = QK - max_vec
                    QK = torch.exp2(QK)

                    max_vec_prev = max_vec_prev - max_vec
                    max_vec_prev = torch.exp2(max_vec_prev)
                    norm_vec = norm_vec * max_vec_prev
                    norm_vec = norm_vec + torch.sum(QK, dim=-1, keepdim=True)

                    # AV
                    o_reg = o_reg * max_vec_prev
                    o_reg = o_reg + torch.matmul(QK, v_block.float())

                # Final normalization
                output[b, h, s_block:s_block+Q_BLOCK_SIZE, :] = (o_reg / norm_vec).to(q_.dtype)

    return output, q_, k_, v_
    
    

def reference_forward(Q, K, V, causal):
    """GQA Reference implementation using BHND layout (batch, heads, seq, dim)"""
    # Convert to float64 and create new leaf tensors with requires_grad
    q_ = Q.detach().to(torch.float32).requires_grad_(True)
    k_ = K.detach().to(torch.float32).requires_grad_(True)
    v_ = V.detach().to(torch.float32).requires_grad_(True)

    # Expand K,V to match Q heads for GQA computation
    group_size = 64 // 8
    # Repeat each KV head group_size times: (B, h_kv, N, D) -> (B, h_q, N, D)
    k_expanded = k_.repeat_interleave(group_size, dim=1)  
    v_expanded = v_.repeat_interleave(group_size, dim=1)

    # Manual pytorch implementation of scaled dot product attention
    QK = torch.matmul(q_, k_expanded.transpose(-2, -1))
    QK /= (q_.size(-1) ** 0.5)

    # Compute LSE before softmax
    lse = torch.logsumexp(QK, dim=-1, keepdim=True)  # (batch, heads, seq, 1)

    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v_expanded).to(torch.bfloat16)

    return output, lse, q_, k_, v_

def simple_flash_backward(Q, K, V, dO, L):
    """GQA version that should match PyTorch exactly"""
    D = Q.shape[-1]
    scale = 1.0 / math.sqrt(D)

    q_ = Q.detach().to(torch.float32).requires_grad_(True)
    k_ = K.detach().to(torch.float32).requires_grad_(True)
    v_ = V.detach().to(torch.float32).requires_grad_(True)
    dO_ = dO.detach().to(torch.float32).requires_grad_(True)
    L_ = L.detach().to(torch.float32).requires_grad_(True)
    
    # Expand K,V to match Q heads for GQA computation  
    K_expanded, V_expanded = expand_kv_for_gqa(k_, v_, h_q, h_kv)

    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(q_, K_expanded.transpose(-2, -1)) * scale
    P = torch.exp(S - L_)
    O = torch.matmul(P, V_expanded)

    # dV - need to sum across grouped heads  
    dV_expanded = torch.matmul(P.transpose(-2, -1), dO_)  # (B, h_q, N, D)
    dV = torch.zeros_like(v_)
    group_size = h_q // h_kv
    for i in range(h_kv):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        dV[:, i, :, :] = dV_expanded[:, start_idx:end_idx, :, :].sum(dim=1)

    # softmax backward
    Delta = (dO_ * O).sum(dim=-1, keepdim=True)                 # (B, h_q, N, 1)
    dS = P * (torch.matmul(dO_, V_expanded.transpose(-2, -1)) - Delta)   # (B, h_q, N, N)

    # chain rule through S = (Q K^T) * scale  
    dQ = torch.matmul(dS, K_expanded) * scale  # (B, h_q, N, D)
    
    # dK - need to sum across grouped heads
    dK_expanded = torch.matmul(dS.transpose(-2, -1), q_) * scale  # (B, h_q, N, D)
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

def read_inputs():
    Q_bhnd = torch.load("inputs/q.pt", map_location='cuda').transpose(1, 2).contiguous().bfloat16()
    K_bhnd = torch.load("inputs/k.pt", map_location='cuda').transpose(1, 2).contiguous().bfloat16()
    V_bhnd = torch.load("inputs/v.pt", map_location='cuda').transpose(1, 2).contiguous().bfloat16()
    dO_bhnd = torch.load("inputs/dO.pt", map_location='cuda').transpose(1, 2).contiguous().bfloat16()
    return Q_bhnd, K_bhnd, V_bhnd, dO_bhnd

def read_prepare_inputs():
    L_tk = torch.load("inputs/L.pt", map_location='cuda').contiguous().float()
    delta_tk = torch.load("inputs/delta.pt", map_location='cuda').contiguous().float()
    O_tk = torch.load("inputs/O.pt", map_location='cuda').contiguous().bfloat16()
    return L_tk, delta_tk, O_tk

# Generate base inputs in BHND format
# Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = generate_inputs()
Q_bhnd, K_bhnd, V_bhnd, dO_bhnd = read_inputs()
# L_tk, delta_tk, O_tk = read_prepare_inputs()

# **************************************************
# Reference forward and backward
# **************************************************
out_ref, lse_ref, q_ref, k_ref, v_ref = reference_forward(Q_bhnd, K_bhnd, V_bhnd, causal)
dQ_ref, dK_ref, dV_ref, delta_ref = simple_flash_backward(Q_bhnd, K_bhnd, V_bhnd, dO_bhnd, lse_ref)

out_ref = out_ref.transpose(1, 2).contiguous()
dQ_ref = dQ_ref.transpose(1, 2).contiguous()
dK_ref = dK_ref.transpose(1, 2).contiguous()
dV_ref = dV_ref.transpose(1, 2).contiguous()

# **************************************************
# AITER forward and backward
# **************************************************

if use_aiter:
    timings = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    print("\nRunning AITER...")

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
    out_aiter_bnhd = out_aiter
    # out_aiter_bhnd = out_aiter.transpose(1, 2)  # BNHD -> BHND
    # q_grad_aiter_bhnd = q_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
    # k_grad_aiter_bhnd = k_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
    # v_grad_aiter_bhnd = v_grad_aiter_bnhd.transpose(1, 2)  # BNHD -> BHND
softmax_lse = softmax_lse.unsqueeze(-1)
# **************************************************
# ThunderKittens
# **************************************************

# Get forwards pass outputs
Q_tk = Q_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True)  
K_tk = K_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True)  
V_tk = V_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True) 
dO_tk = dO_bhnd.transpose(1, 2).bfloat16().clone().contiguous()

# Call TK forward to get O and L
O_tk = torch.zeros_like(out_aiter_bnhd).bfloat16().clone().contiguous()
L_tk = torch.zeros((b, h_q, n, 1), device='cuda').float().transpose(-1, -2).contiguous()

num_print = 8
print("Calling TK forward kernel...")
torch.cuda.synchronize()
tk_kernel_fwd.dispatch_fwd(Q_tk, K_tk, V_tk, O_tk, L_tk)
torch.cuda.synchronize()

if Q_tk.isnan().any():
    print("Q_tk has NaN")
    breakpoint()
if K_tk.isnan().any():
    print("K_tk has NaN")
    breakpoint()
if V_tk.isnan().any():
    print("V_tk has NaN")
    breakpoint()
if O_tk.isnan().any():
    print("O_tk has NaN")
    breakpoint()
if L_tk.isnan().any():
    print("L_tk has NaN")
    breakpoint()
    
print("TK forward completed successfully")
print("TK [O]: ", O_tk[0, 0, :num_print, 0], "Max:", O_tk.max().item())
print("TK [L]: ", L_tk[0, 0, 0, :num_print], "Max:", L_tk.max().item())

# TK
print("Running ThunderKittens ...")
timings = []  # Reset timings for TK
for _ in range(num_iters):
    dQ_tk_in = torch.zeros_like(q_grad_aiter_bnhd).bfloat16().transpose(1, 2).contiguous()
    dQ_tk = torch.zeros_like(q_grad_aiter_bnhd).bfloat16().contiguous()
    dK_tk = torch.zeros_like(k_grad_aiter_bnhd).bfloat16().contiguous()
    dV_tk = torch.zeros_like(v_grad_aiter_bnhd).bfloat16().contiguous()
    # delta_tk = torch.zeros_like(delta_tiled).float()
    delta_tk = torch.zeros((b, h_q, n, 1), device='cuda').float().transpose(-1, -2).contiguous()
    torch.cuda.synchronize()
    start_event.record()

    tk_kernel_bkwd.dispatch_prep(
        O_tk,     # Og
        dO_tk,    # dOg
        delta_tk, # delta
    )
    print("TK [delta]: ", delta_tk[0, 0, 0, :num_print], "Max:", delta_tk.max().item())

    if delta_tk.isnan().any():
        print("delta_tk has NaN")
        breakpoint()
        break

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

    print("TK [dQ_in]: ", dQ_tk_in[0, 0, 0, :num_print], "Max:", dQ_tk_in.max().item())
    print("TK [dK]: ", dK_tk[0, 0, 0, :num_print], "Max:", dK_tk.max().item())
    print("TK [dV]: ", dV_tk[0, 0, 0, :num_print], "Max:", dV_tk.max().item())

    if dQ_tk_in.isnan().any():
        print("dQ_tk_in has NaN")
        breakpoint()
        break
    if dK_tk.isnan().any():
        print("dK_tk has NaN")
        breakpoint()
        break
    if dV_tk.isnan().any():
        print("dV_tk has NaN")
        breakpoint()
        break


    tk_kernel_bkwd.dispatch_dq_shuffle(
        dQ_tk_in,
        dQ_tk
    )

    print("TK [dQ]: ", dQ_tk[0, 0, 0, :num_print], "Max:", dQ_tk.max().item())
    if dQ_tk.isnan().any():
        print("dQ_tk has NaN")
        breakpoint()
        break

    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
    delta_tk = delta_tk.transpose(-1, -2).contiguous()

L_tk = L_tk.transpose(-1, -2).contiguous()

avg_time_tk = sum(timings) / len(timings)
eff_tk = efficiency(flops_ref, avg_time_tk)
print(f"ThunderKittens average execution time: {avg_time_tk:.4f} ms")
print(f"ThunderKittens performance: {eff_tk:.2f} TFLOPS for {b=} h_q={h_q} h_kv={h_kv} {n=} {d=} {causal=}.\n")

# **************************************************
# Comparisons
# **************************************************

print("\nO outputs:")
print("TK: ", O_tk[0, 0, :num_print, 0], "Max:", O_tk.max().item())
print("AITER: ", out_aiter_bnhd[0, 0, :num_print, 0], "Max:", out_aiter_bnhd.max().item())
print("REF: ", out_ref[0, 0, :num_print, 0], "Max:", out_ref.max().item())

print()
print("\nL outputs:")
print("TK: ", L_tk[0, 0, :num_print, 0], "Max:", L_tk.max().item())
print("AITER: ", softmax_lse[0, 0, :num_print, 0], "Max:", softmax_lse.max().item())
print("REF: ", lse_ref[0, 0, :num_print, 0], "Max:", lse_ref.max().item())

print()
print("\nGradient K outputs:")
print("TK: ", dK_tk[0, 0, 0, :num_print], "Max:", dK_tk.max().item())
print("AITER: ", k_grad_aiter_bnhd[0, 0, 0, :num_print], "Max:", k_grad_aiter_bnhd.max().item())
print("REF: ", dK_ref[0, 0, 0, :num_print], "Max:", dK_ref.max().item())

print()
print("Gradient V outputs:")
print("TK: ", dV_tk[0, 0, 0, :num_print], "Max:", dV_tk.max().item())
print("AITER: ", v_grad_aiter_bnhd[0, 0, 0, :num_print], "Max:", v_grad_aiter_bnhd.max().item())
print("REF: ", dV_ref[0, 0, 0, :num_print], "Max:", dV_ref.max().item())

print()
print("Gradient Q outputs:")
print("TK: ", dQ_tk[0, 0, 0, :num_print], "Max:", dQ_tk.max().item())
print("AITER: ", q_grad_aiter_bnhd[0, 0, 0, :num_print], "Max:", q_grad_aiter_bnhd.max().item())
print("REF: ", dQ_ref[0, 0, 0, :num_print], "Max:", dQ_ref.max().item())

# **************************************************
# TK vs REF (robust tolerances & metrics)
# **************************************************
print(f"\nRobustness checks (TK vs REF):") 
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(O_tk, out_ref)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
l_diff, l_err_cnt, l_total, l_rel_error, l_l2_error, l_cos, l_mask = robustness_check(L_tk, lse_ref)
print(f"L: max_abs={l_diff.max().item():.6f}, max_rel={l_rel_error:.4f}, "
      f"rel_l2={l_l2_error:.4f}, cos={l_cos:.6f}, "
      f"errors={l_err_cnt}/{l_total} ({100*l_err_cnt/l_total:.4f}%)")

# **************************************************
# TK vs REF (gradient comparisons)
# **************************************************
print(f"\nGradient comparisons (TK vs REF):") 

q_diff, q_err_cnt, q_total, q_rel_error, q_l2_error, q_cos, q_mask = robustness_check(dQ_tk, dQ_ref)
k_diff, k_err_cnt, k_total, k_rel_error, k_l2_error, k_cos, k_mask = robustness_check(dK_tk, dK_ref)
v_diff, v_err_cnt, v_total, v_rel_error, v_l2_error, v_cos, v_mask = robustness_check(dV_tk, dV_ref)

print(f"Q: max_abs={q_diff.max().item():.6f}, max_rel={q_rel_error:.4f}, "
      f"rel_l2={q_l2_error:.4f}, cos={q_cos:.6f}, "
      f"errors={q_err_cnt}/{q_total} ({100*q_err_cnt/q_total:.4f}%)")
print(f"K: max_abs={k_diff.max().item():.6f}, max_rel={k_rel_error:.4f}, "
      f"rel_l2={k_l2_error:.4f}, cos={k_cos:.6f}, "
      f"errors={k_err_cnt}/{k_total} ({100*k_err_cnt/k_total:.4f}%)")
print(f"V: max_abs={v_diff.max().item():.6f}, max_rel={v_rel_error:.4f}, "
      f"rel_l2={v_l2_error:.4f}, cos={v_cos:.6f}, "
      f"errors={v_err_cnt}/{v_total} ({100*v_err_cnt/v_total:.4f}%)")

# **************************************************
# TK vs AITER (robust tolerances & metrics)
# **************************************************
# Compare O and L with AITER
print(f"\nRobustness checks (TK vs AITER):") 
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(O_tk, out_aiter_bnhd)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
l_diff, l_err_cnt, l_total, l_rel_error, l_l2_error, l_cos, l_mask = robustness_check(L_tk, softmax_lse)
print(f"L: max_abs={l_diff.max().item():.6f}, max_rel={l_rel_error:.4f}, "
      f"rel_l2={l_l2_error:.4f}, cos={l_cos:.6f}, "
      f"errors={l_err_cnt}/{l_total} ({100*l_err_cnt/l_total:.4f}%)")


# **************************************************
# TK vs AITER (gradient comparisons)
# **************************************************
print(f"\nGradient comparisons (TK vs AITER):") 

# Compute diffs in float32 to avoid bf16 quantization in the comparison itself
q_diff, q_err_cnt, q_total, q_rel_error, q_l2_error, q_cos, q_mask = robustness_check(q_grad_aiter_bnhd, dQ_tk)
k_diff, k_err_cnt, k_total, k_rel_error, k_l2_error, k_cos, k_mask = robustness_check(k_grad_aiter_bnhd, dK_tk)
v_diff, v_err_cnt, v_total, v_rel_error, v_l2_error, v_cos, v_mask = robustness_check(v_grad_aiter_bnhd, dV_tk)

print(f"Q grad: max_abs={q_diff.max().item():.6f}, max_rel={q_rel_error:.4f}, "
        f"rel_l2={q_l2_error:.4f}, cos={q_cos:.6f}, "
      f"errors={q_err_cnt}/{q_total} ({100*q_err_cnt/q_total:.4f}%)")
print(f"K grad: max_abs={k_diff.max().item():.6f}, max_rel={k_rel_error:.4f}, "
      f"rel_l2={k_l2_error:.4f}, cos={k_cos:.6f}, "
      f"errors={k_err_cnt}/{k_total} ({100*k_err_cnt/k_total:.4f}%)")
print(f"V grad: max_abs={v_diff.max().item():.6f}, max_rel={v_rel_error:.4f}, "
      f"rel_l2={v_l2_error:.4f}, cos={v_cos:.6f}, "
      f"errors={v_err_cnt}/{v_total} ({100*v_err_cnt/v_total:.4f}%)")

# **************************************************
# REF vs AITER (attention outputs)
# **************************************************
print(f"\nAttention outputs (REF vs AITER):") 
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(O_tk, out_aiter_bnhd)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
l_diff, l_err_cnt, l_total, l_rel_error, l_l2_error, l_cos, l_mask = robustness_check(L_tk, softmax_lse)
print(f"L: max_abs={l_diff.max().item():.6f}, max_rel={l_rel_error:.4f}, "
      f"rel_l2={l_l2_error:.4f}, cos={l_cos:.6f}, "
      f"errors={l_err_cnt}/{l_total} ({100*l_err_cnt/l_total:.4f}%)")

# **************************************************
# Ref vs AITER (gradient comparisons)
# **************************************************
print(f"\nGradient comparisons (REF vs AITER):") 

q_diff, q_err_cnt, q_total, q_rel_error, q_l2_error, q_cos, q_mask = robustness_check(q_grad_aiter_bnhd, dQ_ref)
k_diff, k_err_cnt, k_total, k_rel_error, k_l2_error, k_cos, k_mask = robustness_check(k_grad_aiter_bnhd, dK_ref)
v_diff, v_err_cnt, v_total, v_rel_error, v_l2_error, v_cos, v_mask = robustness_check(v_grad_aiter_bnhd, dV_ref)

print(f"Q grad: max_abs={q_diff.max().item():.6f}, max_rel={q_rel_error:.4f}, "
        f"rel_l2={q_l2_error:.4f}, cos={q_cos:.6f}, "
      f"errors={q_err_cnt}/{q_total} ({100*q_err_cnt/q_total:.4f}%)")
print(f"K grad: max_abs={k_diff.max().item():.6f}, max_rel={k_rel_error:.4f}, "
      f"rel_l2={k_l2_error:.4f}, cos={k_cos:.6f}, "
      f"errors={k_err_cnt}/{k_total} ({100*k_err_cnt/k_total:.4f}%)")
print(f"V grad: max_abs={v_diff.max().item():.6f}, max_rel={v_rel_error:.4f}, "
      f"rel_l2={v_l2_error:.4f}, cos={v_cos:.6f}, "
      f"errors={v_err_cnt}/{v_total} ({100*v_err_cnt/v_total:.4f}%)")