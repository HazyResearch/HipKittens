import torch
import random
import math
import tk_kernel

torch.set_printoptions(
    precision=3,
    sci_mode=False,
    linewidth=220,
    threshold=float("inf")
)

random.seed(0)
torch.manual_seed(0)

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

Q = torch.randn((1, 1, 16, 128), dtype=torch.bfloat16, device='cuda')
dO = torch.randn((1, 1, 16, 128), dtype=torch.bfloat16, device='cuda')
K = torch.randn((1, 1, 256, 128), dtype=torch.bfloat16, device='cuda')
V = torch.randn((1, 1, 256, 128), dtype=torch.bfloat16, device='cuda')
L = torch.randn((1, 1, 1, 16), dtype=torch.float32, device='cuda')
delta = torch.randn((1, 1, 1, 16), dtype=torch.float32, device='cuda')
P = torch.zeros((1, 1, 16, 256), dtype=torch.float32, device='cuda')

# reference
scale_factor = 1.0 / math.sqrt(128)

P = torch.matmul(Q, K.transpose(-1, -2).contiguous()).to(torch.float32) * scale_factor
P -= L.transpose(-1, -2).contiguous()
P = torch.exp(P)

dP = torch.matmul(dO, V.transpose(-1, -2).contiguous()).to(torch.float32)
dP -= delta.transpose(-1, -2).contiguous()
dP = dP * scale_factor
dP = dP * P

dV_ref = torch.matmul(P.transpose(-1, -2).contiguous().to(torch.bfloat16), dO)
dK_ref = torch.matmul(dP.transpose(-1, -2).contiguous().to(torch.bfloat16), Q)
dQ_ref = torch.matmul(dP.to(torch.bfloat16), K)

# tk
dV_tk = torch.zeros_like(dV_ref)
dK_tk = torch.zeros_like(dK_ref)

dQ_tk_tmp = torch.zeros_like(dQ_ref)
dQ_tk = torch.zeros_like(dQ_ref)

torch.cuda.synchronize()
tk_kernel.dispatch_micro(Q, dO, K, V, L, delta, dK_tk, dV_tk, dQ_tk_tmp)
tk_kernel.dispatch_dq_shuffle(dQ_tk_tmp, dQ_tk)
torch.cuda.synchronize()

# check
num_print = 8
print("dV_tk: ", dV_tk[0, 0, 0, 0:num_print], "Max:", dV_tk.max().item())
print("dV_ref: ", dV_ref[0, 0, 0, 0:num_print], "Max:", dV_ref.max().item())
print("dK_tk: ", dK_tk[0, 0, 0, 0:num_print], "Max:", dK_tk.max().item())
print("dK_ref: ", dK_ref[0, 0, 0, 0:num_print], "Max:", dK_ref.max().item())
print("dQ_tk: ", dQ_tk[0, 0, 0, 0:num_print], "Max:", dQ_tk.max().item())
print("dQ_ref: ", dQ_ref[0, 0, 0, 0:num_print], "Max:", dQ_ref.max().item())


dV_diff, dV_err_cnt, dV_total, dV_rel_error, dV_l2_error, dV_cos, dV_mask = robustness_check(dV_ref, dV_tk)
dK_diff, dK_err_cnt, dK_total, dK_rel_error, dK_l2_error, dK_cos, dK_mask = robustness_check(dK_ref, dK_tk)
dQ_diff, dQ_err_cnt, dQ_total, dQ_rel_error, dQ_l2_error, dQ_cos, dQ_mask = robustness_check(dQ_ref, dQ_tk)
print(f"dV: max_abs={dV_diff.max().item():.6f}, max_rel={dV_rel_error:.4f}, "
      f"rel_l2={dV_l2_error:.4f}, cos={dV_cos:.6f}, "
      f"errors={dV_err_cnt}/{dV_total} ({100*dV_err_cnt/dV_total:.4f}%)")
print(f"dK: max_abs={dK_diff.max().item():.6f}, max_rel={dK_rel_error:.4f}, "
      f"rel_l2={dK_l2_error:.4f}, cos={dK_cos:.6f}, "
      f"errors={dK_err_cnt}/{dK_total} ({100*dK_err_cnt/dK_total:.4f}%)")
print(f"dQ: max_abs={dQ_diff.max().item():.6f}, max_rel={dQ_rel_error:.4f}, "
      f"rel_l2={dQ_l2_error:.4f}, cos={dQ_cos:.6f}, "
      f"errors={dQ_err_cnt}/{dQ_total} ({100*dQ_err_cnt/dQ_total:.4f}%)")