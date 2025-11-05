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

n = 256
d = 128

# pytorch
x = torch.ones((1, n, 1, d), dtype=torch.bfloat16, device='cuda')
x[0, :32, 0, :] = 0
x[0, 32:64, 0, :] = 1
x[0, 64:96, 0, :] = 2
x[0, 96:128, 0, :] = 3

# reference
y = x

# tk
y_tk = torch.zeros_like(y)
tk_kernel.dispatch_micro(x, y_tk)

# check
diff = (y - y_tk).abs().max()
print(y.shape, x.shape)
print(f"diff: {diff}")

print(y[0, 0:32, 0, :1].T)
print(y_tk[0, 0:32, 0, :1].T)

print(y[0, 32:64, 0, :1].T)
print(y_tk[0, 32:64, 0, :1].T)

print(y[0, 64:96, 0, :1].T)
print(y_tk[0, 64:96, 0, :1].T)

print(y[0, 96:128, 0, :1].T)
print(y_tk[0, 96:128, 0, :1].T)
