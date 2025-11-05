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

kv_seq = 256
qo_seq = 16

# pytorch
x = torch.randn((1, 1, kv_seq, qo_seq), dtype=torch.bfloat16, device='cuda')


# reference
y = x

# tk
y_tk = torch.zeros_like(y)
tk_kernel.dispatch_micro(x, y_tk)

# check
diff = (y - y_tk).abs().max()
print(y.shape, x.shape)
print(f"diff: {diff}")

print(y[0, 0, :16, :1].T)
print(y_tk[0, 0, :16, :1].T)

print(y[0, 0, 16:32, :1].T)
print(y_tk[0, 0, 16:32, :1].T)

print(y[0, 0, 32:48, :1].T)
print(y_tk[0, 0, 32:48, :1].T)

print(y[0, 0, 48:64, :1].T)
print(y_tk[0, 0, 48:64, :1].T)
