"""Shared helpers for the epilogue correctness tests.

Inputs are small integers times a power of two, so the fp32 GEMM is exact in any summation order:
the kernel and the reference start from identical numbers, and the only differences a test allows
are the ones it names.
"""
import importlib
import math
import os
import pathlib
import subprocess
import sys

import torch

EPI = pathlib.Path(__file__).resolve().parents[1]   # kernels/cdna4/gemm/bf16fp32/epilogue
BASE = EPI.parent                                    # kernels/cdna4/gemm/bf16fp32 (upstream base GEMM)
DEV = "cuda"
if str(EPI) not in sys.path:
    sys.path.insert(0, str(EPI))                    # so tests can `import hk`

SHAPES = [(256, 256, 128), (4096, 4096, 128), (8192, 2048, 4096), (4096, 4096, 4096)]  # (M, N, K)
BAD_SHAPES = [(384, 256, 128), (256, 256, 192)]                                         # M % 256, K % 128


def hk_shape_error(M, N, K):
    """(exception, message) hk raises for a BAD_SHAPES case: hk.prepare (Python) checks K, launch (C++) checks M."""
    if K % 128:
        return ValueError, "K % 128 == 0"
    return RuntimeError, "M and N must be multiples of 256"


_modules = {}


def module(name):
    """Import tk_<name> ("base" = upstream's tk_kernel), building it first unless EPILOGUE_PREBUILT=1."""
    if name not in _modules:
        directory, make_args, import_name = (
            (BASE, [], "tk_kernel") if name == "base" else (EPI, [f"KERNEL={name}"], f"tk_{name}"))
        if os.environ.get("EPILOGUE_PREBUILT") != "1":
            subprocess.run(["make", *make_args, "clean"], cwd=directory, check=True, capture_output=True)
            build = subprocess.run(["make", *make_args], cwd=directory, capture_output=True, text=True)
            if build.returncode != 0:
                raise RuntimeError(f"make {' '.join(make_args)} failed:\n{build.stderr[-3000:]}")
        if str(directory) not in sys.path:
            sys.path.insert(0, str(directory))
        _modules[name] = importlib.import_module(import_name)
    return _modules[name]


def stream():
    """torch's current stream, as the trailing argument every tk_<name>.dispatch takes."""
    return torch.cuda.current_stream().cuda_stream


def weight(b, layout="plain", gamma=None):
    """hk-prepared weight from a test operand b [N, K] (hk.prepare takes the natural [K, N])."""
    import hk
    return hk.prepare(b.t().contiguous(), layout=layout, gamma=gamma)


def pow2_gamma(K, seed=7):
    """bf16 gamma of powers of two (0.5, 1, 2), so folding it into b keeps the GEMM exact."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (2.0 ** torch.randint(-1, 2, (K,), generator=g, device=DEV).float()).to(torch.bfloat16)


def inputs(M, N, K, kind, seed=0):
    """bf16 a [M, K] and b [N, K] whose product is exact in fp32. kind: "random", "large" or "zeros".

    Entries are integers in [-2, 2]; a is scaled by 2^exp. Each product is a multiple of 2^exp with
    magnitude <= 4, so any partial sum is at most 4K units of 2^exp, far below fp32's 2^24.
    random: exp = -round(log2(2 sqrt K)) gives outputs with standard deviation ~1.
    large:  exp = 3 gives outputs ~1e3.
    """
    if kind == "zeros":
        return (torch.zeros(M, K, dtype=torch.bfloat16, device=DEV),
                torch.zeros(N, K, dtype=torch.bfloat16, device=DEV))
    g = torch.Generator(device=DEV).manual_seed(seed)
    ia = torch.randint(-2, 3, (M, K), generator=g, device=DEV)
    ib = torch.randint(-2, 3, (N, K), generator=g, device=DEV)
    exp = -round(math.log2(2 * math.sqrt(K))) if kind == "random" else 3
    return (ia.float() * 2.0 ** exp).to(torch.bfloat16), ib.float().to(torch.bfloat16)


def gemm_exact(a, b):
    """a @ b.T in fp64 (exact for inputs() data), returned as fp32 (still exact)."""
    return (a.double() @ b.double().T).float()


def bf16_step(x):
    """Spacing between adjacent bf16 values at |x|; 0 where x == 0. x is an fp32 tensor."""
    return torch.where(x == 0, torch.zeros_like(x), torch.exp2(torch.floor(torch.log2(x.abs())) - 7))


def _report(bad, got, want, what):
    i = tuple(bad.nonzero()[0].tolist())
    return (f"{int(bad.sum())} of {bad.numel()} elements {what}; "
            f"first at {i}: got {got[i].item()!r}, want {want[i].item()!r}")


def to_bf16_chopped(x):
    """fp32 -> bf16 the way HipKittens' store does it: keep the top 16 bits (round toward zero).
    torch's .to(torch.bfloat16) rounds to nearest instead, which differs by one bf16 step."""
    return (x.float().contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


def silu_kernel(x):
    """SiLU as the kernel computes it: torch's SiLU, except exactly 0 where sigmoid(x) is below fp32's
    smallest normal (2^-126, x < about -87), because the hardware reciprocal returns 0 there."""
    return torch.where(torch.sigmoid(x) < 2.0 ** -126, x * 0, torch.nn.functional.silu(x))


def assert_bitexact(got, ref):
    """bf16 got must equal fp32 ref stored the way the kernel stores it (to_bf16_chopped)."""
    want = to_bf16_chopped(ref)
    bad = got != want
    assert not bad.any(), _report(bad, got, want, "differ")


def assert_within_bf16_step(got, ref):
    """bf16 got may differ from fp32 ref stored as the kernel does by at most one bf16 step
    (0 where ref is 0)."""
    want = to_bf16_chopped(ref).float()
    bad = (got.float() - want).abs() > bf16_step(want)
    assert not bad.any(), _report(bad, got.float(), want, "differ by more than 1 bf16 step")


def assert_rel_close(got, ref, rel):
    """|got - ref| <= rel * |ref| elementwise (so ref == 0 requires got == 0)."""
    bad = (got - ref).abs() > rel * ref.abs()
    assert not bad.any(), _report(bad, got, ref, f"exceed relative error {rel:.2e}")
