import torch
import random
import math
import tk_kernel_causal_bkwd
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


causal = True
b = 8
h_q = 16  # number of query heads  
h_kv = 8  # number of key/value heads (for GQA)
group_size = h_q // h_kv  # queries per KV head group
n = 2048
d = 128
dtype = torch.bfloat16
mean = 10
std = 0.1 

num_warmup = 50
num_iters = 100

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)

# **************************************************
# Benchmarking
# **************************************************

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


def simple_flash_backward(Q, K, V, dO, m, l, causal=False):
    """Simple version that should match PyTorch exactly"""
    D = Q.shape[-1]
    N = Q.shape[-2]  # sequence length
    scale = 1.0 / math.sqrt(D)
    
    h_q = Q.shape[1]
    h_kv = K.shape[1]
    group_size = h_q // h_kv

    # Expand K,V to match Q heads for GQA computation
    K_expanded, V_expanded = expand_kv_for_gqa(K, V, h_q, h_kv)
    # Recompute scores and probabilities with saved m, l
    S = torch.matmul(Q, K_expanded.transpose(-2, -1)) * scale
    
    # Apply causal mask if needed
    if causal:
        causal_mask = torch.triu(torch.ones(N, N, device=S.device, dtype=S.dtype), diagonal=1).bool()
        S = S.masked_fill(causal_mask, float('-inf'))
    
    P = torch.exp(S - m.unsqueeze(-1)) / l.unsqueeze(-1)
    # Zero out masked positions to avoid nan
    if causal:
        P = P.masked_fill(causal_mask, 0.0)

    # breakpoint()
    
    O = torch.matmul(P, V_expanded)

    # dV - need to sum across grouped heads
    dV_expanded = torch.matmul(P.transpose(-2, -1), dO) # (B, hq, N, D)
    dV = torch.zeros_like(V)
    for i in range(h_kv):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size
        dV[:, i, :, :] = dV_expanded[:, start_idx:end_idx, :, :].sum(dim=1)

    # softmax backward
    Delta = (dO * O).sum(dim=-1, keepdim=True)   # (B, hq, N, 1)
    dS = P * (torch.matmul(dO, V_expanded.transpose(-2, -1)) - Delta) # (B, hq, N, N)
    # Apply causal mask to gradients
    if causal:
        dS = dS.masked_fill(causal_mask, 0.0)

    # chain rule through S = (Q K^T) * scale
    dQ = torch.matmul(dS, K_expanded) * scale # (B, hq, N, D)
    # dK - need to sum across grouped heads
    dK_expanded = torch.matmul(dS.transpose(-2, -1), Q) * scale # (B, hq, N, D)
    dK = torch.zeros_like(K)
    for i in range(h_kv):
        start_idx = i * group_size
        end_idx = (i + 1) * group_size 
        dK[:, i, :, :] = dK_expanded[:, start_idx:end_idx, :, :].sum(dim=1)

    return dQ, dK, dV, Delta


# **************************************************
# Generate inputs
# **************************************************


def generate_tensor(shape, mean, std, dtype, device):
    tensor = torch.randn(shape, dtype=dtype, device=device)
    magnitude = torch.norm(tensor, dim=-1, keepdim=True)
    scaled_tensor = tensor * (torch.randn(magnitude.shape, dtype=dtype, device=device) * std + mean) / magnitude
    return scaled_tensor.contiguous()

def generate_inputs():
    # Generate in BHND layout (batch, heads, seq, dim)
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
flops_ref = flops(b, n, h_q, d, causal)

# **************************************************
# AITER forward and backward
# **************************************************

for _ in range(num_warmup):
    Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
    K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
    V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
    dO_aiter = dO_bhnd.transpose(1, 2).contiguous()
    out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal=causal, return_lse=True, deterministic=False)
    out_aiter.backward(dO_aiter)
timings_ref = []
for _ in range(num_iters):
    Q_aiter = Q_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
    K_aiter = K_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
    V_aiter = V_bhnd.transpose(1, 2).contiguous().detach().requires_grad_(True)  
    dO_aiter = dO_bhnd.transpose(1, 2).contiguous()
    out_aiter, softmax_lse = aiter.flash_attn_func(Q_aiter, K_aiter, V_aiter, causal=causal, return_lse=True, deterministic=False)
    torch.cuda.synchronize()
    start_event.record()
    out_aiter.backward(dO_aiter)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_ref.append(elapsed_time)

print(f"{out_aiter.dtype=}")
avg_time_ref = sum(timings_ref) / len(timings_ref)
eff_ref = efficiency(flops_ref, avg_time_ref)
print(f"AITER (AMD) reference average execution time: {avg_time_ref:.4f} ms")
print(f"AITER (AMD) reference performance: {eff_ref:.2f} TFLOPS for {b=} {h_q=} {h_kv=} {n=} {d=} {causal=}.\n")

q_grad_aiter_bnhd = Q_aiter.grad
k_grad_aiter_bnhd = K_aiter.grad  
v_grad_aiter_bnhd = V_aiter.grad
out_aiter_bnhd = out_aiter

# **************************************************
# Tiled Reference
# **************************************************

Q_tiled = Q_bhnd.clone().contiguous().detach().requires_grad_(True)  
K_tiled = K_bhnd.clone().contiguous().detach().requires_grad_(True)  
V_tiled = V_bhnd.clone().contiguous().detach().requires_grad_(True)  
K_expanded = K_tiled.repeat_interleave(h_q // h_kv, dim=1)
V_expanded = V_tiled.repeat_interleave(h_q // h_kv, dim=1)
dO_tiled = dO_bhnd.clone().contiguous()  
QK = torch.matmul(Q_tiled.float(), K_expanded.transpose(-2, -1).float()) / math.sqrt(d)
if causal:
    N = QK.shape[-1]  # sequence length
    causal_mask = torch.triu(torch.ones(N, N, device=QK.device, dtype=torch.bool), diagonal=1)
    QK = QK.masked_fill(causal_mask, float('-inf'))
m_tiled = QK.max(dim=-1, keepdim=True)[0] 
exp_scores = torch.exp(QK - m_tiled)  
l_tiled = exp_scores.sum(dim=-1, keepdim=True)  
P_tiled = exp_scores / l_tiled
O_tiled = torch.matmul(P_tiled, V_expanded.float())
m_tiled = m_tiled.squeeze(-1)
l_tiled = l_tiled.squeeze(-1)

# Pass ORIGINAL K_tiled and V_tiled (both with h_kv heads)
dQ_tiled, dK_tiled, dV_tiled, delta_tiled = simple_flash_backward(
    Q_tiled.float(), 
    K_tiled.float(),  # Original with h_kv heads
    V_tiled.float(),  # Original with h_kv heads
    dO_tiled.float(), 
    m_tiled, 
    l_tiled, 
    causal=causal
)
out_tiled_bhnd = O_tiled
q_grad_tiled_bhnd = dQ_tiled


# **************************************************
# ThunderKittens
# **************************************************

# Get forwards pass outputs
Q_tk = Q_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True)  
K_tk = K_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True)  
V_tk = V_bhnd.transpose(1, 2).bfloat16().clone().contiguous().detach().requires_grad_(True) 

L_logsumexp = m_tiled + torch.log(l_tiled) 
L_tk = L_logsumexp.float().unsqueeze(-1).contiguous()
check = softmax_lse.float().unsqueeze(-1).contiguous()
diff = (check - L_tk).abs()
print(f"L diff: {diff.max().item():.6f}")

# Call TK forward to get O and L
dO_tk = dO_bhnd.transpose(1, 2).bfloat16().clone().contiguous()
O_tk = O_tiled.bfloat16().transpose(1, 2).clone().contiguous()

# TK
print("Running ThunderKittens ...")
dQ_tk_in = torch.zeros_like(q_grad_aiter_bnhd).bfloat16().transpose(1, 2).contiguous()
dQ_tk = torch.zeros_like(q_grad_aiter_bnhd).bfloat16().contiguous()
dK_tk = torch.zeros_like(k_grad_aiter_bnhd).bfloat16().contiguous()
dV_tk = torch.zeros_like(v_grad_aiter_bnhd).bfloat16().contiguous()
delta_tk = torch.zeros((b, h_q, n, 1), device='cuda').float().transpose(-1, -2).contiguous()

print("Running ThunderKittens ...")
timings = []
for _ in range(num_warmup):
    tk_kernel_causal_bkwd.dispatch_prep(
        O_tk,     # Og
        dO_tk,    # dOg
        delta_tk, # delta
    )

    tk_kernel_causal_bkwd.dispatch_bwd_combined(
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

    tk_kernel_causal_bkwd.dispatch_dq_shuffle(
        dQ_tk_in,
        dQ_tk
    )

for _ in range(num_iters):
    dQ_tk_in = torch.zeros_like(q_grad_aiter_bnhd).bfloat16().transpose(1, 2).contiguous()
    dQ_tk = torch.zeros_like(q_grad_aiter_bnhd).bfloat16().contiguous()
    dK_tk = torch.zeros_like(k_grad_aiter_bnhd).bfloat16().contiguous()
    dV_tk = torch.zeros_like(v_grad_aiter_bnhd).bfloat16().contiguous()
    delta_tk = torch.zeros((b, h_q, n, 1), device='cuda').float().transpose(-1, -2).contiguous()
    torch.cuda.synchronize()
    start_event.record()

    tk_kernel_causal_bkwd.dispatch_prep(
        O_tk,     # Og
        dO_tk,    # dOg
        delta_tk, # delta
    )

    tk_kernel_causal_bkwd.dispatch_bwd_combined(
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

    tk_kernel_causal_bkwd.dispatch_dq_shuffle(
        dQ_tk_in,
        dQ_tk
    )

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

dQ_simple = dQ_tiled.transpose(1, 2).contiguous()
dK_simple = dK_tiled.transpose(1, 2).contiguous()
dV_simple = dV_tiled.transpose(1, 2).contiguous()

num_print = 8
num_print_seq = 1

# TK vs AITER
print(f"\nTK vs AITER comparison:")
print(f"\nO outputs ({O_tk.shape}):")
print("TK: ", O_tk[0, 0, :num_print, 0], "Max:", O_tk.max().item())
print("AITER: ", out_aiter_bnhd[0, 0, :num_print, 0], "Max:", out_aiter_bnhd.max().item())
print("Simple: ", O_tiled[0, 0, :num_print, 0], "Max:", O_tiled.max().item())

print()
print(f"\nGradient K outputs ({dK_tk.shape}):")
print("TK: ", dK_tk[0, 0:num_print_seq, 0, :num_print], "Max:", dK_tk.max().item())
print("AITER: ", k_grad_aiter_bnhd[0, 0:num_print_seq, 0, :num_print], "Max:", k_grad_aiter_bnhd.max().item())
print("Simple: ", dK_simple[0, 0:num_print_seq, 0, :num_print], "Max:", dK_simple.max().item())

print()
print(f"Gradient V outputs ({dV_tk.shape}):")
print("TK: ", dV_tk[0, 0:num_print_seq, 0, :num_print], "Max:", dV_tk.max().item())
print("AITER: ", v_grad_aiter_bnhd[0, 0:num_print_seq, 0, :num_print], "Max:", v_grad_aiter_bnhd.max().item())
print("Simple: ", dV_simple[0, 0:num_print_seq, 0, :num_print], "Max:", dV_simple.max().item())

print()
print(f"Gradient Q outputs ({dQ_tk.shape}):")
print("TK: ", dQ_tk[0, 0:num_print_seq, 0, :num_print], "Max:", dQ_tk.max().item())
print("AITER: ", q_grad_aiter_bnhd[0, 0:num_print_seq, 0, :num_print], "Max:", q_grad_aiter_bnhd.max().item())
print("Simple: ", dQ_simple[0, 0:num_print_seq, 0, :num_print], "Max:", dQ_simple.max().item())

# **************************************************
# TK vs AITER (robust tolerances & metrics)
# **************************************************
# Compare O and L with AITER
print(f"\nRobustness checks (TK vs AITER):") 
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(O_tk, out_aiter_bnhd)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")

# **************************************************
# Gradient comparisons
# **************************************************
print(f"\nTK vs AITER:") 
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

print(f"\nAITER vs. simple comparison:")
q_diff, q_err_cnt, q_total, q_rel_error, q_l2_error, q_cos, q_mask = robustness_check(q_grad_aiter_bnhd, dQ_simple)
k_diff, k_err_cnt, k_total, k_rel_error, k_l2_error, k_cos, k_mask = robustness_check(k_grad_aiter_bnhd, dK_simple)
v_diff, v_err_cnt, v_total, v_rel_error, v_l2_error, v_cos, v_mask = robustness_check(v_grad_aiter_bnhd, dV_simple)
print(f"Q grad: max_abs={q_diff.max().item():.6f}, max_rel={q_rel_error:.4f}, "
      f"rel_l2={q_l2_error:.4f}, cos={q_cos:.6f}, "
      f"errors={q_err_cnt}/{q_total} ({100*q_err_cnt/q_total:.4f}%)")
print(f"K grad: max_abs={k_diff.max().item():.6f}, max_rel={k_rel_error:.4f}, "
      f"rel_l2={k_l2_error:.4f}, cos={k_cos:.6f}, "
      f"errors={k_err_cnt}/{k_total} ({100*k_err_cnt/k_total:.4f}%)")
print(f"V grad: max_abs={v_diff.max().item():.6f}, max_rel={v_rel_error:.4f}, "
      f"rel_l2={v_l2_error:.4f}, cos={v_cos:.6f}, "
      f"errors={v_err_cnt}/{v_total} ({100*v_err_cnt/v_total:.4f}%)")


print(f"\nTK vs. simple comparison:")
q_diff, q_err_cnt, q_total, q_rel_error, q_l2_error, q_cos, q_mask = robustness_check(dQ_tk, dQ_simple)
k_diff, k_err_cnt, k_total, k_rel_error, k_l2_error, k_cos, k_mask = robustness_check(dK_tk, dK_simple)
v_diff, v_err_cnt, v_total, v_rel_error, v_l2_error, v_cos, v_mask = robustness_check(dV_tk, dV_simple)
print(f"Q grad: max_abs={q_diff.max().item():.6f}, max_rel={q_rel_error:.4f}, "
      f"rel_l2={q_l2_error:.4f}, cos={q_cos:.6f}, "
      f"errors={q_err_cnt}/{q_total} ({100*q_err_cnt/q_total:.4f}%)")
print(f"K grad: max_abs={k_diff.max().item():.6f}, max_rel={k_rel_error:.4f}, "
      f"rel_l2={k_l2_error:.4f}, cos={k_cos:.6f}, "
      f"errors={k_err_cnt}/{k_total} ({100*k_err_cnt/k_total:.4f}%)")
print(f"V grad: max_abs={v_diff.max().item():.6f}, max_rel={v_rel_error:.4f}, "
      f"rel_l2={v_l2_error:.4f}, cos={v_cos:.6f}, "
      f"errors={v_err_cnt}/{v_total} ({100*v_err_cnt/v_total:.4f}%)")

batch = 1
# print(f"{q_diff[batch,:4].max().item()=}") 
# print(f"{q_diff[batch,:256].max().item()=}") 
# print(f"{q_diff[batch,256:512].max().item()=}") 
# print(f"{q_diff[batch,512:768].max().item()=}") 
# print(f"{q_diff[batch,768:1024].max().item()=}") 

# # print location of max error for q_diff
# max_error_idx = q_diff.max(dim=1)[1]
# print(f"{q_diff[batch,max_error_idx[batch]].max().item()=}") 


