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

# pytorch
x = torch.randn((1, 1, 1024, 128), dtype=torch.bfloat16, device='cuda')
y = x

# tk
y_tk = torch.zeros_like(y)
tk_kernel.dispatch_micro(x, y_tk)

# check
diff = (y - y_tk).abs().max()
num_diffs = (y - y_tk).abs().nonzero().shape[0]
print(y.shape, x.shape)
print(f"max diff: {diff}")
print(f"num diffs: {num_diffs}")