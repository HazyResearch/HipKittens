import torch
import triton
import triton.language as tl

import tk_running
import tk_pow2

M, N, K = 8192, 8192, 8192
BLOCK = 128
KB = K // BLOCK
NB = N // BLOCK

SEED = 100
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
dev = "cuda"

TOL = {
    torch.float32:  dict(rtol=1.3e-6, atol=3e-4),
    torch.bfloat16: dict(rtol=1.6e-2, atol=3e-4),
}


def make_inputs(is_1d2d):
    A = (torch.randn(M, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    B = (torch.randn(N, K, device=dev) * 0.5).to(torch.float8_e4m3fn)
    scale_a = (0.5 + torch.rand(KB, M, device=dev)).contiguous()
    if is_1d2d:
        scale_b = (0.5 + torch.rand(NB, KB, device=dev)).contiguous()
    else:
        scale_b = (0.5 + torch.rand(KB, N, device=dev)).contiguous()
    return A, B, scale_a, scale_b


def pow2_round(x):
    bits = x.view(torch.int32)
    exp = ((bits >> 23) & 0xFF).to(torch.int32)
    out = (exp << 23).view(torch.float32)
    return out


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


def reference(A, B, scale_a, scale_b, is_1d2d, pow2):
    sa = scale_a
    sb = scale_b
    if pow2:
        sa = pow2_round(sa)
        sb = pow2_round(sb)
    one = torch.tensor(1.0, dtype=torch.float32, device=dev)
    acc = torch.zeros(M, N, dtype=torch.float32, device=dev)
    for kb in range(KB):
        a_blk = A[:, kb * BLOCK:(kb + 1) * BLOCK].contiguous()
        b_blk = B[:, kb * BLOCK:(kb + 1) * BLOCK].contiguous()
        partial = torch._scaled_mm(a_blk, b_blk.t(), scale_a=one, scale_b=one,
                                   out_dtype=torch.float32, use_fast_accum=False)
        sa_k = sa[kb, :].unsqueeze(1)
        if is_1d2d:
            sb_k = sb[:, kb].repeat_interleave(BLOCK).unsqueeze(0)
        else:
            sb_k = sb[kb, :].unsqueeze(0)
        partial = partial * sa_k
        fma_(acc, partial, sb_k.expand_as(partial).contiguous())
    return acc


def benchmark(fn, A, B, c, scale_a, scale_b, warmup=20, iters=100):
    for _ in range(warmup):
        fn(A, B, c, scale_a, scale_b)
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn(A, B, c, scale_a, scale_b)
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    best = min(times)
    flops = 2.0 * M * N * K
    return flops / (best * 1e-3) / 1e12


KERNELS = {
    ("running", "1d2d", "fp32"): (tk_running.gemm_1d2d, torch.float32, True, False),
    ("running", "1d1d", "fp32"): (tk_running.gemm_1d1d, torch.float32, False, False),
    ("running", "1d2d", "bf16"): (tk_running.gemm_1d2d_bf16, torch.bfloat16, True, False),
    ("running", "1d1d", "bf16"): (tk_running.gemm_1d1d_bf16, torch.bfloat16, False, False),
    ("pow2", "1d2d", "fp32"): (tk_pow2.gemm_pow2_1d2d, torch.float32, True, True),
    ("pow2", "1d1d", "fp32"): (tk_pow2.gemm_pow2_1d1d, torch.float32, False, True),
    ("pow2", "1d2d", "bf16"): (tk_pow2.gemm_pow2_1d2d_bf16, torch.bfloat16, True, True),
    ("pow2", "1d1d", "bf16"): (tk_pow2.gemm_pow2_1d1d_bf16, torch.bfloat16, False, True),
}


def run(variant, mode, otype):
    fn, dtype, is_1d2d, pow2 = KERNELS[(variant, mode, otype)]

    A, B, scale_a, scale_b = make_inputs(is_1d2d)

    C = torch.zeros(M, N, dtype=dtype, device=dev)
    fn(A, B, C, scale_a, scale_b)
    torch.cuda.synchronize()

    ref = reference(A, B, scale_a, scale_b, is_1d2d, pow2).to(dtype)

    tol = TOL[dtype]
    diff = (C.to(torch.float32) - ref.to(torch.float32)).abs()
    max_abs = diff.max().item()
    ok = torch.allclose(C, ref, **tol)
    bad = (~torch.isclose(C, ref, **tol)).sum().item()

    tflops = benchmark(fn, A, B, C, scale_a, scale_b)

    print(f"=== {variant} {mode} {otype} : {M}x{N}x{K} ===")
    print(f"  TFLOPS: {tflops:.1f}")
    print(f"  rtol: {tol['rtol']:.1e} atol: {tol['atol']:.1e}, max_abs_err: {max_abs:.2e}, bad: {bad} / {M*N}, {'PASSED' if ok else 'FAILED'}")
    return ok


if __name__ == "__main__":
    ok = True
    for variant in ("running", "pow2"):
        for mode in ("1d2d", "1d1d"):
            for otype in ("fp32", "bf16"):
                ok &= run(variant, mode, otype)
    print("\n" + ("ALL PASSED" if ok else "SOME FAILED"))
