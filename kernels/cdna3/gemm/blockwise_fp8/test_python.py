import torch
import triton
import triton.language as tl
import tk_kernel

M, N, K = 8192, 8192, 8192
BLOCK = 128
KB = K // BLOCK
NB = N // BLOCK

torch.manual_seed(42)
dev = "cuda"


@triton.jit
def fma_kernel(y_ptr, x_ptr, s_ptr, n_elem, BLK: tl.constexpr = 1024):
    pid = tl.program_id(0)
    idx = pid * BLK + tl.arange(0, BLK)
    mask = idx < n_elem
    y = tl.load(y_ptr + idx, mask=mask)
    x = tl.load(x_ptr + idx, mask=mask)
    s = tl.load(s_ptr + idx, mask=mask)
    tl.store(y_ptr + idx, tl.fma(x, s, y), mask=mask)


def fma_(y, x, s):
    n = y.numel()
    grid = ((n + 1023) // 1024,)
    fma_kernel[grid](y, x, s, n)
    return y


def make_inputs(is_1d2d):
    act = (torch.randn(M, K, device=dev) * 0.5).to(torch.float8_e4m3fnuz)
    wt = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fnuz)
    scale_a = (0.5 + torch.rand(KB, M, device=dev)).contiguous()
    if is_1d2d:
        scale_b = (0.5 + torch.rand(NB, KB, device=dev)).contiguous()
    else:
        scale_b = (0.5 + torch.rand(KB, N, device=dev)).contiguous()
    return act, wt, scale_a, scale_b


def reference(act, wt, scale_a, scale_b, is_1d2d):
    af = act.to(torch.float32)
    wf = wt.to(torch.float32)
    acc = torch.zeros(M, N, dtype=torch.float32, device=dev)
    for kb in range(KB):
        a_blk = af[:, kb * BLOCK:(kb + 1) * BLOCK]
        w_blk = wf[:, kb * BLOCK:(kb + 1) * BLOCK]
        partial = a_blk @ w_blk.t()
        sa = scale_a[kb, :].unsqueeze(1)
        if is_1d2d:
            sb = scale_b[:, kb].repeat_interleave(BLOCK).unsqueeze(0)
        else:
            sb = scale_b[kb, :].unsqueeze(0)
        # kernel does: Cacc = fma(partial, sa*sb, Cacc)
        s = (sa * sb).expand_as(partial).contiguous()
        fma_(acc, partial, s)
    return acc


def benchmark(fn, act, wt, c, scale_a, scale_b, warmup=20, iters=100):
    for _ in range(warmup):
        fn(act, wt, c, scale_a, scale_b)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(act, wt, c, scale_a, scale_b)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    best = min(times)
    avg = sum(times) / len(times)
    flops = 2.0 * M * N * K
    return best, avg, flops / (best * 1e-3) / 1e12, flops / (avg * 1e-3) / 1e12


KERNELS = {
    ("1d2d", "fp32"): (tk_kernel.gemm_1d2d, torch.float32, True),
    ("1d1d", "fp32"): (tk_kernel.gemm_1d1d, torch.float32, False),
    ("1d2d", "bf16"): (tk_kernel.gemm_1d2d_bf16, torch.bfloat16, True),
    ("1d1d", "bf16"): (tk_kernel.gemm_1d1d_bf16, torch.bfloat16, False),
}


def run(mode, otype):
    fn, dtype, is_1d2d = KERNELS[(mode, otype)]
    act, wt, scale_a, scale_b = make_inputs(is_1d2d)
    c = torch.zeros(M, N, dtype=dtype, device=dev)
    fn(act, wt, c, scale_a, scale_b)
    torch.cuda.synchronize()

    ref = reference(act, wt, scale_a, scale_b, is_1d2d)
    if dtype == torch.bfloat16:
        ref = ref.to(torch.bfloat16)
    cf = c.to(torch.float32)
    rf = ref.to(torch.float32)
    diff = (cf - rf).abs()
    max_abs = diff.max().item()
    ok = torch.allclose(c, ref, atol=0, rtol=0)
    bad = (~torch.isclose(c, ref, atol=0, rtol=0)).sum().item()
    crit = "exact(atol=0,rtol=0)"

    best_ms, avg_ms, best_tflops, avg_tflops = benchmark(fn, act, wt, c, scale_a, scale_b)

    print(f"=== {mode} {otype} : {M}x{N}x{K} ===")
    print(f"  {crit}, max_abs_err: {max_abs:.2e}, bad: {bad} / {M*N}")
    print(f"  TFLOPS: best {best_tflops:.1f}, avg {avg_tflops:.1f}   (best {best_ms:.3f} ms)")
    print(f"  {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    ok = True
    ok &= run("1d2d", "fp32")
    ok &= run("1d1d", "fp32")
    ok &= run("1d2d", "bf16")
    ok &= run("1d1d", "bf16")
    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
