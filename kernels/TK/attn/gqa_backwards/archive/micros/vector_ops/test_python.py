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

b = 1
h = 1
n = 32
d = 32

# pytorch
x = torch.randn((b, h, n, d), dtype=torch.float32, device='cuda')
vec = torch.randn((b, h, d, 1), dtype=torch.float32, device='cuda')
# y = x.sum(dim=-1, keepdim=True)
y = x - vec

# tk
y_tk = torch.zeros_like(y)
tk_kernel.dispatch_micro(x, vec, y_tk)

# check
diff = (y - y_tk).abs().max()
print(y.shape, vec.shape, x.shape)
print(f"diff: {diff}")


print(y[0, 0, 0:2, :8])
print(y_tk[0, 0, 0:2, :8])
print()

print(y[0, 0, 3:6, :8])
print(y_tk[0, 0, 3:6, :8])