"""vLLM's unquantized linear layers on the generated GEMM, where it is faster.

GEMM is 57.6% of Qwen3-8B's device time at batch 16 / context 4096
(tools/hk-bench/vllm_profile.py), and on gfx1100 every one of those GEMMs ends
in `torch.nn.functional.linear` -- hipBLASLt. The installed vLLM's skinny-GEMM
path is gated on gfx9, and correctly: its `wvSplitK` is compiled for gfx9 only
and fails a device-side assert on gfx1100.

The routing rule is measured on the model's own shapes against exactly that
call (tools/hk-bench/vllm_gemm.py; one process, interleaved, every variant
checked against fp32 before it is timed):

    M >= 1024                         hk GEMM on every shape, 1.05-1.23x
    384 <= M < 1024, M % 128 == 0     hk GEMM, 1.09-1.43x
    1 <= M <= 64                      hk decode GEMM, 1.47-2.04x per step
    anything else                     vLLM's own path

The gap between the first two rows is a comb, not a curve. The last M block
backs up to end at M and recomputes a full 128 rows, so between multiples of
128 the kernel does extra work, and at small grids two workgroups then share a
WGP: M=192 measured 0.71-0.95x. Above 1024 the wasted rows are at most an
eighth of the work, which the kernel's lead covers.

Decode is the third row, and a different problem: M is the batch, the math is
free, and the whole cost is streaming the weights once. `ops.decode_gemm` is
the kernel for that; timed the way a step reads the weights -- 36 distinct
layers back to back -- it streams 663-802 GB/s at M <= 16 against hipBLASLt's
317-638. A batch is padded to whole 16-row token tiles: the kernel stores
whole tiles and loads them without a predicate, so a shorter activation would
be read past its end.

How it gets in. `UnquantizedLinearMethod.apply` calls
`dispatch_unquantized_gemm()` on every forward, as does the LM head, so
rebinding that one name -- by identity, in every vLLM module that imported it
-- routes every unquantized linear without touching a model file.

Why a custom op rather than a Python function. Under vLLM's torch.compile the
token count is a symbolic shape, so branching on it in traced code is a guard
and a recompile. An opaque op with a fake implementation keeps the branch out
of the graph; it is what vLLM does for its own `rocm_unquantized_gemm`, and it
launches on torch's stream, so CUDA-graph capture sees it.

    HK_PATCH_VLLM=1 HK_VLLM_GEMM=1 vllm serve Qwen/Qwen3-8B
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from . import _common

PREFIXES = ("vllm",)

#: The smallest M routed whatever its alignment, and the smallest routed when
#: it is a whole number of 128-row blocks.
M_ANY = 1024
M_ALIGNED = 384

#: What the GEMM is compiled for. On anything else the op routes nothing.
ARCH = "gfx1100"

#: Which way each call went. Python counters, so a CUDA-graph replay -- which
#: runs no Python -- is not counted; they describe eager and capture.
STATS: Dict[str, int] = {"hk": 0, "hk_decode": 0, "fallback": 0}

_ORIG_DISPATCH: Optional[Callable] = None
_FALLBACK: Optional[Callable] = None
_LIB = None
_OPS: Dict[str, Callable] = {}
_ARCH_OK: Dict[int, bool] = {}


def route(m: int, n: int, k: int, *, bf16: bool = True, bias: bool = False,
          contiguous_weight: bool = True) -> bool:
    """Whether y = x @ W.T with x (m, k) and W (n, k) goes to the generated GEMM.

    Pure, so the rule is tested without a GPU. A bias falls back rather than
    paying a second pass over the output; no Qwen3 linear has one.
    """
    from ..ops import gemm as hk_gemm  # noqa: PLC0415

    if not bf16 or bias or not contiguous_weight:
        return False
    if n % hk_gemm.BLOCK_N or k % hk_gemm.K_STEP:
        return False
    if m >= M_ANY:
        return True
    return m >= M_ALIGNED and m % hk_gemm.BLOCK_M == 0


def route_decode(m: int, n: int, k: int, *, bf16: bool = True,
                 bias: bool = False, contiguous_weight: bool = True) -> bool:
    """Whether a decode-sized GEMM goes to `ops.decode_gemm`. Pure, like `route`."""
    from ..ops import decode_gemm  # noqa: PLC0415

    if not bf16 or bias or not contiguous_weight:
        return False
    return decode_gemm.fits(m, n, k)


def _arch_ok(device) -> bool:
    idx = device.index if device.index is not None else 0
    ok = _ARCH_OK.get(idx)
    if ok is None:
        import torch  # noqa: PLC0415

        name = torch.cuda.get_device_properties(idx).gcnArchName or ""
        ok = _ARCH_OK[idx] = name.split(":")[0] == ARCH
    return ok


def _op_for(m: int, n: int, k: int) -> Callable:
    """The torch.ops registration for the schedule this shape gets."""
    from ..ops import gemm as hk_gemm  # noqa: PLC0415
    from ..runtime.torch_ext import torch_op  # noqa: PLC0415

    s = hk_gemm.schedule_for(m, n, k)
    key = hk_gemm._sched_name({**hk_gemm.DEFAULT, **s})
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = torch_op(hk_gemm.TUNER._kernel_for(s))
    return op


def _decode_op(s: Dict[str, Any]) -> Callable:
    from ..ops import decode_gemm  # noqa: PLC0415
    from ..runtime.torch_ext import torch_op  # noqa: PLC0415

    key = "decode_" + "_".join(f"{k}{v}" for k, v in sorted(s.items()))
    op = _OPS.get(key)
    if op is None:
        op = _OPS[key] = torch_op(decode_gemm.kernel_for(**s))
    return op


def _fallback(x, weight, bias):
    """What vLLM would have run: its own op where it has one, else F.linear."""
    global _FALLBACK
    if _FALLBACK is None:
        import torch  # noqa: PLC0415

        try:
            _FALLBACK = torch.ops.vllm.rocm_unquantized_gemm
        except (AttributeError, RuntimeError):
            _FALLBACK = torch.nn.functional.linear
    return _FALLBACK(x, weight, bias)


def linear(x, weight, bias=None):
    """y = x @ weight.T (+ bias), on the generated GEMM where `route` says."""
    import torch  # noqa: PLC0415

    k = x.shape[-1]
    m = x.numel() // k if k else 0
    n = weight.shape[0]
    how = dict(bf16=x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16,
               bias=bias is not None, contiguous_weight=weight.is_contiguous())
    if route(m, n, k, **how) and _arch_ok(x.device):
        x2 = x.reshape(m, k)
        if not x2.is_contiguous():
            x2 = x2.contiguous()
        out = torch.empty((m, n), device=x.device, dtype=x.dtype)
        _op_for(m, n, k)(x2, weight, out)
        STATS["hk"] += 1
        return out.view(*x.shape[:-1], n)
    if route_decode(m, n, k, **how) and _arch_ok(x.device):
        from ..ops import decode_gemm  # noqa: PLC0415

        s = decode_gemm.schedule_for(m, k)
        rows = 16 * s["tokens"]
        x2 = x.reshape(m, k)
        if m < rows:
            x2 = torch.nn.functional.pad(x2, (0, 0, 0, rows - m))
        elif not x2.is_contiguous():
            x2 = x2.contiguous()
        out = torch.empty((rows, n), device=x.device, dtype=x.dtype)
        _decode_op(s)(x2, weight, out)
        STATS["hk_decode"] += 1
        return out[:m].view(*x.shape[:-1], n)
    STATS["fallback"] += 1
    return _fallback(x, weight, bias)


def _linear_fake(x, weight, bias=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


def register_op():
    """Define `torch.ops.hk_vllm.linear`. Idempotent."""
    global _LIB
    if _LIB is not None:
        return
    import torch  # noqa: PLC0415

    lib = torch.library.Library("hk_vllm", "FRAGMENT")
    lib.define("linear(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor")
    lib.impl("linear", linear, "CUDA")
    torch.library.register_fake("hk_vllm::linear", _linear_fake, lib=lib)
    _LIB = lib


def _gemm(layer, x, weight, bias=None):
    """What `dispatch_unquantized_gemm()` hands back once patched."""
    import torch  # noqa: PLC0415

    if not x.is_cuda:
        return torch.nn.functional.linear(x, weight, bias)
    return torch.ops.hk_vllm.linear(x, weight, bias)


def dispatch_unquantized_gemm(*args, **kwargs):
    """Stands in for vLLM's. The arguments -- a linear-backend name in newer
    releases -- select among CUDA backends and mean nothing on ROCm."""
    return _gemm


def apply() -> str:
    """Route vLLM's linear layers. Returns what happened, in one line."""
    global _ORIG_DISPATCH
    if not _common.importable("vllm"):
        return "vllm is not importable; linear not patched"
    if _ORIG_DISPATCH is not None:
        return "linear: already patched"
    from vllm.model_executor.layers import utils as vu  # noqa: PLC0415
    from vllm.platforms import current_platform  # noqa: PLC0415

    if not current_platform.is_rocm():
        return "linear: not patched -- not a ROCm platform"
    register_op()
    _ORIG_DISPATCH = vu.dispatch_unquantized_gemm
    names = _common.rebind(PREFIXES, _ORIG_DISPATCH, dispatch_unquantized_gemm)
    from ..ops import decode_gemm  # noqa: PLC0415

    return (f"linear -> hk where M >= {M_ANY}, or M >= {M_ALIGNED} in whole "
            f"128-row blocks, or M <= {16 * decode_gemm.MAX_TOKENS} on the "
            f"decode kernel ({len(names)} binding"
            f"{'' if len(names) == 1 else 's'} rebound)")


def revert() -> str:
    global _ORIG_DISPATCH
    if _ORIG_DISPATCH is None:
        return "linear: not patched"
    names = _common.rebind(PREFIXES, dispatch_unquantized_gemm, _ORIG_DISPATCH)
    _ORIG_DISPATCH = None
    return f"linear -> vllm ({len(names)} rebound)"


def status() -> str:
    return "patched" if _ORIG_DISPATCH is not None else "not patched"
