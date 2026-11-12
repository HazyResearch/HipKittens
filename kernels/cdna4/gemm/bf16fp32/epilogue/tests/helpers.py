"""Helpers shared by the epilogue tests.

module(name)  builds `make KERNEL=<name>` and imports tk_<name> ("base" = upstream tk_kernel);
              EPILOGUE_PREBUILT=1 skips the build.
inputs(...)   bf16 a, b whose fp32 product is exact, so a test can name the only differences it allows:
              the bf16 store (assert_bitexact vs to_bf16_chopped), one bf16 step (assert_within_bf16_step),
              or a stated relative error (assert_rel_close, assert_norm_close).
SHAPES / BAD_SHAPES  shapes every test runs, and shapes hk must reject (hk_shape_error gives the message).
"""
import importlib
import math
import os
import pathlib
import subprocess
import sys

import torch

EPI = pathlib.Path(__file__).resolve().parents[1]   # kernels/cdna4/gemm/bf16fp32/epilogue
BASE = EPI.parent                                    # base GEMM
DEV = "cuda"

SHAPES = [(256, 256, 128), (4096, 4096, 128), (8192, 2048, 4096), (4096, 4096, 4096)]  # (M, N, K)
BAD_SHAPES = [(384, 256, 128), (256, 256, 192)]                                         # M % 256, K % 128


def hk_shape_error(M, N, K):
    """(exception, message) hk raises for a BAD_SHAPES case."""
    if K % 128:
        return ValueError, "K % 128 == 0"
    return ValueError, "M % 256 == 0"


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
    """Current stream handle for tk_<name>.dispatch."""
    return torch.cuda.current_stream().cuda_stream


def weight(b, gamma=None):
    """b, or b * gamma over K (stands in for an input carrying gamma: (a * gamma) @ b.T == a @ (b * gamma).T)."""
    return b if gamma is None else (b.float() * gamma.float()[None, :]).to(torch.bfloat16)


def pow2_gamma(K, seed=7):
    """bf16 gamma in {0.5, 1, 2}: keeps the GEMM exact."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (2.0 ** torch.randint(-1, 2, (K,), generator=g, device=DEV).float()).to(torch.bfloat16)


def general_gamma(n, seed=7):
    """bf16 gamma in [0.5, 1.5): not powers of two, so rounding order is observable."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    return (torch.rand(n, generator=g, device=DEV) + 0.5).to(torch.bfloat16)


def inputs(M, N, K, kind, seed=0):
    """bf16 a [M, K], b [N, K] with an fp32-exact product. kind: "random" (outputs ~1), "large" (~1e3), "zeros".

    Integers in [-2, 2], a scaled by 2^exp: partial sums stay below 4K units of 2^exp << 2^24.
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
    """Exact a @ b.T as fp32."""
    return (a.double() @ b.double().T).float()


def bf16_step(x):
    """bf16 spacing at |x| (0 at 0)."""
    return torch.where(x == 0, torch.zeros_like(x), torch.exp2(torch.floor(torch.log2(x.abs())) - 7))


def _report(bad, got, want, what):
    i = tuple(bad.nonzero()[0].tolist())
    return (f"{int(bad.sum())} of {bad.numel()} elements {what}; "
            f"first at {i}: got {got[i].item()!r}, want {want[i].item()!r}")


def to_bf16_chopped(x):
    """fp32 -> bf16 by truncation, as HipKittens' store does (torch rounds to nearest)."""
    return (x.float().contiguous().view(torch.int32) & -65536).view(torch.float32).to(torch.bfloat16)


def silu_kernel(x):
    """torch SiLU, but 0 where sigmoid(x) < 2^-126 (the hardware reciprocal flushes)."""
    return torch.where(torch.sigmoid(x) < 2.0 ** -126, x * 0, torch.nn.functional.silu(x))


def assert_bitexact(got, ref):
    """got == to_bf16_chopped(ref)."""
    want = to_bf16_chopped(ref)
    bad = got != want
    assert not bad.any(), _report(bad, got, want, "differ")


def assert_within_bf16_step(got, ref):
    """got within one bf16 step of to_bf16_chopped(ref)."""
    want = to_bf16_chopped(ref).float()
    bad = (got.float() - want).abs() > bf16_step(want)
    assert not bad.any(), _report(bad, got.float(), want, "differ by more than 1 bf16 step")


def assert_rel_close(got, ref, rel):
    """|got - ref| <= rel * |ref| elementwise."""
    bad = (got - ref).abs() > rel * ref.abs()
    assert not bad.any(), _report(bad, got, ref, f"exceed relative error {rel:.2e}")


def assert_norm_close(got, want, rel):
    """||got - want|| <= rel * ||want||."""
    err = ((got.double() - want.double()).norm() / want.double().norm()).item()
    assert err <= rel, f"normwise relative error {err:.2e} > {rel:.0e}"
