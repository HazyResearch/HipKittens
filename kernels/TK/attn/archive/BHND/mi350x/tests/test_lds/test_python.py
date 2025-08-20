import torch
import tk_kernel
import random

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
D = 32
dtype = torch.bfloat16
q = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
k = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
v = torch.randn(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)

k_out = torch.zeros(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)
v_out = torch.zeros(B, H, N, D, dtype=dtype, device='cuda', requires_grad=True)

tk_kernel.dispatch_micro(q, k, v, k_out, v_out)

k_out_ref = k.clone()
v_out_ref = v.clone()

print("Out")
print(k_out)
print("Ref")
print(k_out_ref)

diff = k_out.float() - k_out_ref.float()
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

    