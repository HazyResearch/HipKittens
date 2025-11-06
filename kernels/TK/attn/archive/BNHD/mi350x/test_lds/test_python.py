import torch
import tk_kernel
import tk_golden
import random
from aiter.ops.triton.mha import flash_attn_func

torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=220,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

# Inputs
B = 1
H = 1
N = 64
D = 64
dtype = torch.bfloat16
q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
k = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
v = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)

# out_ref_aiter = flash_attn_func(q, k, v)
out_ref = torch.zeros(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
out = torch.zeros(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
tk_golden.dispatch_micro(q, k, v, out_ref)
tk_kernel.dispatch_micro(q, k, v, out)

print("Out")
print(out)
print("Out ref")
print(out_ref)

diff = out.float() - out_ref.float()
print("Diff")
print(diff)

max_error = diff.max().item()
mean_error = diff.mean().item()
error_count = (diff > 0.1).sum().item()

print(f"Max error: {max_error}")
print(f"Mean error: {mean_error}")
print(f"Error count: {error_count}")