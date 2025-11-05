import torch
import tk_fwd_causal_kernel
import random
from torch.nn.functional import scaled_dot_product_attention
import aiter

torch.set_printoptions(
    precision=3,        # decimals
    sci_mode=False,     # True → always scientific
    linewidth=220,      # characters per line before folding
    threshold=float("inf")  # print every element, no summarising "..."
)

# Inputs
B = 8
H = 16
H_KV = 8
N = 2048
D = 128
causal = True
dtype = torch.bfloat16

num_warmup = 100
num_iters = 100

def flops(batch, seqlen, nheads, headdim, causal, mode="fwd"):
    """Calculate FLOPs for attention operation."""
    flop = 4 * batch * seqlen**2 * nheads * headdim // (2 if causal else 1)
    return flop

def efficiency(flop, time):
    """Calculate efficiency in TFLOPS."""
    flop = flop / 1e12  # convert to TFLOPS
    time = time / 1e3   # convert to seconds
    return flop / time

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

start_event = torch.cuda.Event(enable_timing=True) # in milliseconds
end_event = torch.cuda.Event(enable_timing=True)
flops_ref = flops(B, N, H, D, causal)

# Reference matmul using AITER
for _ in range(num_warmup):
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    out_ref, lse_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
timings_ref = []
torch.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    torch.cuda.synchronize()
    start_event.record()
    out_ref, lse_ref = aiter.flash_attn_func(q, k, v, causal=causal, return_lse=True, deterministic=True)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings_ref.append(elapsed_time)
print(f"{out_ref.dtype=}")
avg_time_ref = sum(timings_ref) / len(timings_ref)
eff_ref = efficiency(flops_ref, avg_time_ref)
print(f"AITER (AMD) reference average execution time: {avg_time_ref:.4f} ms")
print(f"AITER (AMD) reference performance: {eff_ref:.2f} TFLOPS for {B=} {H=} {N=} {D=} {causal=}.\n")


# Kernel matmul
for _ in range(num_warmup):
    out = torch.zeros(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda', requires_grad=True)
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    tk_fwd_causal_kernel.dispatch_fwd(q, k, v, out, lse)
timings = []
out = torch.zeros(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
lse = torch.zeros(B, H, 1, N, dtype=torch.float32, device='cuda', requires_grad=True)
torch.manual_seed(0)
random.seed(0)
for _ in range(num_iters):
    q = torch.randn(B, N, H, D, dtype=dtype, device='cuda', requires_grad=True)
    k = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    v = torch.randn(B, N, H_KV, D, dtype=dtype, device='cuda', requires_grad=True)
    torch.cuda.synchronize()
    start_event.record()
    tk_fwd_causal_kernel.dispatch_fwd(q, k, v, out, lse)
    end_event.record()
    torch.cuda.synchronize()
    elapsed_time = start_event.elapsed_time(end_event)
    timings.append(elapsed_time)
print(f"{out.dtype=}")
avg_time = sum(timings) / len(timings)
eff = efficiency(flops_ref, avg_time)
print(f"Average execution time: {avg_time:.4f} ms")
print(f"Performance: {eff:.2f} TFLOPS for {N}x{N} matrix multiplication.\n")

# Compare against reference
# num_print = 8
# print(f"TK vs AITER comparison:")
# print("\nO outputs:")
# print("TK: ", out[0, 0, :num_print, 0], "Max:", out.max().item())
# print("AITER: ", out_ref[0, 0, :num_print, 0], "Max:", out_ref.max().item())

# print("\nLSE outputs:")
# print("TK: ", lse[0, 0, 0, :num_print], "Max:", lse.max().item())
# print("AITER: ", lse_ref[0, 0, :num_print], "Max:", lse_ref.max().item())

print("\nRobustness check:")
o_diff, o_err_cnt, o_total, o_rel_error, o_l2_error, o_cos, o_mask = robustness_check(out, out_ref)
print(f"O: max_abs={o_diff.max().item():.6f}, max_rel={o_rel_error:.4f}, "
      f"rel_l2={o_l2_error:.4f}, cos={o_cos:.6f}, "
      f"errors={o_err_cnt}/{o_total} ({100*o_err_cnt/o_total:.4f}%)")
l_diff, l_err_cnt, l_total, l_rel_error, l_l2_error, l_cos, l_mask = robustness_check(lse, lse_ref.unsqueeze(-1).transpose(-1, -2))
print(f"LSE: max_abs={l_diff.max().item():.6f}, max_rel={l_rel_error:.4f}, "
      f"rel_l2={l_l2_error:.4f}, cos={l_cos:.6f}, "
      f"errors={l_err_cnt}/{l_total} ({100*l_err_cnt/l_total:.4f}%)")


# O-DIFFs


warp_0_diff = o_diff[:, :512, :, 0:]
print(f"Warp 0 diff: {warp_0_diff.max().item():.6f}")

warp_1_diff = o_diff[:, 512:1024, :, 0:]
print(f"Warp 1 diff: {warp_1_diff.max().item():.6f}")


warp_2_diff = o_diff[:, 1024:1536, :, 0:]
print(f"Warp 2 diff: {warp_2_diff.max().item():.6f}")

warp_3_diff = o_diff[:, 1536:2048, :, 0:]
print(f"Warp 3 diff: {warp_3_diff.max().item():.6f}")

# warp_0_diff = o_diff[:, :32, :, 0:]
# print(f"Warp 0 diff: {warp_0_diff.max().item():.6f}")

# warp_1_diff = o_diff[:, 32:64, :, 0:]
# print(f"Warp 1 diff: {warp_1_diff.max().item():.6f}")

# warp_2_diff = o_diff[:, 64:96, :, 0:]
# print(f"Warp 2 diff: {warp_2_diff.max().item():.6f}")

# warp_3_diff = o_diff[:, 96:128, :, 0:]
# print(f"Warp 3 diff: {warp_3_diff.max().item():.6f}")

# warp_4_diff = o_diff[:, 128:160, :, 0:]
# print(f"Warp 4 diff: {warp_4_diff.max().item():.6f}")

# warp_5_diff = o_diff[:, 160:192, :, 0:]
# print(f"Warp 5 diff: {warp_5_diff.max().item():.6f}")

# warp_6_diff = o_diff[:, 192:224, :, 0:]
# print(f"Warp 6 diff: {warp_6_diff.max().item():.6f}")

# warp_7_diff = o_diff[:, 224:256, :, 0:]
# print(f"Warp 7 diff: {warp_7_diff.max().item():.6f}")


# start = 0

# warp_8_diff = o_diff[:, 256:288, :, start:]
# print(f"Warp 0 diff: {warp_8_diff.max().item():.6f}")

# warp_9_diff = o_diff[:, 288:320, :, start:]
# print(f"Warp 1 diff: {warp_9_diff.max().item():.6f}")

# warp_10_diff = o_diff[:, 320:352, :, start:]
# print(f"Warp 2 diff: {warp_10_diff.max().item():.6f}")

# warp_11_diff = o_diff[:, 352:384, :, start:]
# print(f"Warp 3 diff: {warp_11_diff.max().item():.6f}")

# warp_12_diff = o_diff[:, 384:416, :, start:]
# print(f"Warp 4 diff: {warp_12_diff.max().item():.6f}")

# warp_13_diff = o_diff[:, 416:448, :, start:]
# print(f"Warp 5 diff: {warp_13_diff.max().item():.6f}")

# warp_14_diff = o_diff[:, 448:480, :, start:]
# print(f"Warp 6 diff: {warp_14_diff.max().item():.6f}")

# warp_15_diff = o_diff[:, 480:512, :, start:]
# print(f"Warp 7 diff: {warp_15_diff.max().item():.6f}")

# breakpoint()

# LSE-DIFFs





