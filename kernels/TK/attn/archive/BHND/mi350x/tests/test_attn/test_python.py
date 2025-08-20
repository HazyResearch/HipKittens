import torch
import tk_kernel_row
import tk_kernel_col
import random
import time
import math
from torch.nn.functional import scaled_dot_product_attention
from aiter.ops.triton.mha import flash_attn_func

torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=120,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

# Inputs
B = 1
H = 1
N = 32
D = 32
causal = False
dtype = torch.bfloat16
q = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
k = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
v = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)

attention_block_ref = torch.zeros(B, H, N, N, dtype=dtype, device='cuda', requires_grad=True)
attention_block = torch.zeros(B, H, N, N, dtype=dtype, device='cuda', requires_grad=True)

# out_ref_pytorch = scaled_dot_product_attention(q, k, v, is_causal=causal)

# Kernel matmul
out_ref = torch.zeros(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
out = torch.zeros(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
tk_kernel_row.dispatch_micro(q, k, v, attention_block, out)
tk_kernel_col.dispatch_micro(q, k, v, attention_block_ref, out_ref)

# print("Out")
# print(out)
# print("Ref")
# print(out_ref)
attention_block_diff = attention_block - attention_block_ref
print(f"attention_block_diff")
print(attention_block_diff)

diff = out - out_ref
print(f"diff")
print(diff)

# print(f"diff[4:8]")
# print(diff[4:8])

# print(f"diff[8:12]")
# print(diff[8:12])

# print()
# print(f"diff[12:16, 0:8]")
# print(diff[12:16, 0:8])

# print()
# print(f"diff[16:20, 0:8]")
# print(diff[16:20,0:8])


max_diff = diff.abs().max()
print(f"Max diff: {max_diff}")

max_attention_block_diff = attention_block_diff.abs().max()
print(f"Max attention_block_diff: {max_attention_block_diff}")

    