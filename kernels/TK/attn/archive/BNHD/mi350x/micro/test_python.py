import torch
import tk_kernel
import random

profiling = True
profiling_ref = False
torch.manual_seed(0)
random.seed(0)

torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=220,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

# Inputs
ROWS = 32
COLS = 128

A = torch.randn(ROWS, COLS, dtype=torch.bfloat16, device='cuda') / 10.0  
C = torch.zeros(ROWS, COLS, dtype=torch.bfloat16, device='cuda')

tk_kernel.dispatch_micro(A, C)

# C_ref = torch.matmul(A.t(), A).float()
C_ref = A.float()

print("Out")
print(C)
print("Ref")
print(C_ref)

diff = C.float() - C_ref.float()
print("Diff")
print(diff)

max_diff = diff.abs().max()
print(f"Max diff: {max_diff}")


