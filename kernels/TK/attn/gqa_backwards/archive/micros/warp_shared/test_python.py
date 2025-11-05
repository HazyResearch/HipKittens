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
n = 128
d = 32

# pytorch
x = torch.ones((b, h, n, d), dtype=torch.bfloat16, device='cuda')
x[:, :, :32, :] = 0
x[:, :, 32:64, :] = 1
x[:, :, 64:96, :] = 2
x[:, :, 96:128, :] = 3
y = x

# tk
y_tk = torch.zeros_like(y)
tk_kernel.dispatch_micro(x, y_tk)

# check
diff = (y - y_tk).abs().max()
print(y.shape, x.shape)
print(f"diff: {diff}")

print(y[0, 0, 0:32, :1].T)
print(y_tk[0, 0, 0:32, :1].T)

print(y[0, 0, 32:64, :1].T)
print(y_tk[0, 0, 32:64, :1].T)

print(y[0, 0, 64:96, :1].T)
print(y_tk[0, 0, 64:96, :1].T)

print(y[0, 0, 96:128, :1].T)
print(y_tk[0, 0, 96:128, :1].T)
