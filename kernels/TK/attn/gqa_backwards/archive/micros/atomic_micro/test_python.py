import torch
import random
import tk_kernel

torch.set_printoptions(
    precision=2,        
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
x = torch.randn((b, h, n, d), dtype=torch.bfloat16, device='cuda')
print(x.shape)
y = x[:, :, 0:32, :] + x[:, :, 32:64, :] + x[:, :, 64:96, :] + x[:, :, 96:128, :]
print(y.shape)

# tk
y_tk = torch.zeros_like(y)
tk_kernel.dispatch_micro(x, y_tk)

# check
diff = (y - y_tk).abs()
max_diff = diff.max()
print(y.shape, x.shape)
print(f"max diff: {max_diff}")

print(diff[0, 0, :1, :16])


