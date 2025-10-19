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

def robustness_check(ref, pred):
    ref = ref.float()
    pred = pred.float()
    diff = (ref - pred).abs()
    denom = ref.abs().clamp_min(1e-6)
    mask = (diff > (0.001 + 0.05 * denom))
    error_count = mask.sum().item()
    numel = ref.numel()
    rel_error = error_count / numel
    l2_error = (diff.pow(2).sum().sqrt() / ref.pow(2).sum().sqrt()).item()
    cos = torch.nn.functional.cosine_similarity(ref.flatten(), pred.flatten(), dim=0).item()
    return diff, error_count, numel, rel_error, l2_error, cos, mask 

n = 32
d = 128

# pytorch
x = torch.randn((1, n, 1, d), dtype=torch.bfloat16, device='cuda')

# reference
y = x

# tk
y_tk = torch.zeros_like(y)
tk_kernel.dispatch_micro(x, y_tk)

# check
diff = (y - y_tk).abs().max()
print(y.shape, x.shape)
print(f"diff: {diff}")

num_print = 8
print("y: ", y[0, 0:num_print, 0, :num_print])
print("y_tk: ", y_tk[0, 0:num_print, 0, :num_print])

diff, error_count, numel, rel_error, l2_error, cos, mask = robustness_check(y, y_tk)
print(f"A: max_abs={diff.max().item():.6f}, max_rel={rel_error:.4f}, "
      f"rel_l2={l2_error:.4f}, cos={cos:.6f}, "
      f"errors={error_count}/{numel} ({100*error_count/numel:.4f}%)")
