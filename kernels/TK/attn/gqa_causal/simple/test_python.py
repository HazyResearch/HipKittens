import torch
import tk_kernel
import random
import aiter

torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=220,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

# Inputs
B = 1
H = 1
N = 32
D = 64
causal = True
dtype = torch.bfloat16

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

# Reference matmul using AITER
torch.manual_seed(0)
random.seed(0)
q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
k = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
v = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
out_ref, lse_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)

# Kernel matmul
torch.manual_seed(0)
random.seed(0)
q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
k = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
v = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
out = torch.zeros(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
tk_kernel.dispatch_micro(q, k, v, out)

# Compare against reference
num_print = 16
print(f"\n TK vs AITER comparison:")
print("\nO outputs:")
print("TK: ", out[0, 0, :num_print, 0], "Max:", out.max().item())
print("AITER: ", out_ref[0, 0, :num_print, 0], "Max:", out_ref.max().item())

print("Robustness check:")
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(out, out_ref)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")


# breakpoint()
