"""vLLM's unquantized linear layers on the generated GEMMs, where they are faster.

GEMM is 57.6% of Qwen3-8B's device time at batch 16 / context 4096
(tools/hk-bench/vllm_profile.py), and on gfx1100 every one of those GEMMs ends
in `torch.nn.functional.linear` -- hipBLASLt. The installed vLLM's skinny-GEMM
path is gated on gfx9, and correctly: its `wvSplitK` is compiled for gfx9 only
and fails a device-side assert on gfx1100.

Where each call goes is a measured table, not a rule:
`hk/tuned/routes_vllm_linear_<arch>.json`, written by
tools/hk-bench/route_tune.py, which times the engine's path, the prefill GEMM
and the decode GEMV on each (N, K) of the model at every M up to 64, every 16
rows to 1024 and coarser above -- each gated against fp32 first -- and routes
an M to a kernel only where it won by 3%. `$HK_VLLM_ROUTES` names another
table. A shape the table has never seen takes vLLM's path, as does every
shape on an arch with no table. See `engine.RouteTable` for how an M between
two measured ones is answered.

The table replaces four hand-written lines because the GEMM's time is a comb
in M, not a curve: the last 128-row block is computed whole however little of
it is kept, so between multiples of 128 the kernel does extra work, and at
small grids two workgroups share a WGP. Thresholds either route the teeth or
give up the gaps.

Decode is a different problem: M is the batch, the math is free, and the
whole cost is streaming the weights once. `ops.decode_gemm` is the kernel for
that, at 1.93-2.03x hipBLASLt per step at M <= 32.

How it gets in. `UnquantizedLinearMethod.apply` calls
`dispatch_unquantized_gemm()` on every forward, as does the LM head, so
rebinding that one name -- by identity, in every vLLM module that imported it
-- routes every unquantized linear without touching a model file.

Why a custom op rather than a Python function. Under vLLM's torch.compile the
token count is a symbolic shape, so branching on it in traced code is a guard
and a recompile. An opaque op with a fake implementation keeps the branch out
of the graph; it is what vLLM does for its own `rocm_unquantized_gemm`, and it
launches on torch's stream, so CUDA-graph capture sees it. Its route depends
on shapes alone, which a graph captured per batch size fixes, so it declares
itself graph-safe -- and `engine` holds it to that under capture.

    HK_PATCH_VLLM=1 HK_VLLM_GEMM=1 vllm serve Qwen/Qwen3-8B
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from . import _common, engine

PREFIXES = ("vllm",)

#: Where a routing table override comes from.
ROUTES_ENV = "HK_VLLM_ROUTES"

_ORIG_DISPATCH: Optional[Callable] = None
_FALLBACK: Optional[Callable] = None
_OPS: Dict[str, Callable] = {}
_ARCH_OK: Dict[int, bool] = {}
_TABLE: list = []


def _linear_fake(x, weight, bias=None):
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


PATCH = engine.Op("hk_vllm::linear",
                  "linear(Tensor x, Tensor weight, Tensor? bias=None) -> Tensor",
                  _linear_fake, paths=("hk", "hk_decode", "fallback"), graph="safe")

#: Which way each call went; see `engine.Patch` for what a replay counts as.
STATS = PATCH.stats


def routes() -> Optional[engine.RouteTable]:
    """The routing table for this process's arch, or None. Loaded once."""
    if not _TABLE:
        from ..target import current_arch  # noqa: PLC0415

        _TABLE.append(engine.shipped_table("vllm_linear", current_arch(), ROUTES_ENV))
    return _TABLE[0]


def set_routes(table: Optional[engine.RouteTable]) -> None:
    """Route by `table` from now on; None for the shipped one."""
    if any(PATCH.captured.values()):
        import warnings  # noqa: PLC0415

        warnings.warn("hk_vllm::linear: routes changed after a CUDA graph was "
                      "captured; graphs captured before keep replaying the old "
                      "routes", stacklevel=2)
    _TABLE.clear()
    if table is not None:
        _TABLE.append(table)


def route_for(m: int, n: int, k: int, *, bf16: bool = True, bias: bool = False,
              contiguous_weight: bool = True) -> str:
    """"hk", "hk_decode" or "fallback" for y = x @ W.T, x (m, k), W (n, k).

    The table says which kernel measured fastest; this also asks whether that
    kernel takes the call at all. A bias falls back rather than paying a
    second pass over the output; no Qwen3 linear has one.
    """
    from ..ops import decode_gemm  # noqa: PLC0415
    from ..ops import gemm as hk_gemm  # noqa: PLC0415

    if m < 1 or not bf16 or bias or not contiguous_weight:
        return "fallback"
    table = routes()
    path = table.lookup(m, n, k) if table is not None else "fallback"
    if path == "hk" and (n % hk_gemm.BLOCK_N or k % hk_gemm.K_STEP):
        return "fallback"
    if path == "hk_decode" and not decode_gemm.fits(m, n, k):
        return "fallback"
    return path if path in PATCH.paths else "fallback"


def route(m: int, n: int, k: int, **how) -> bool:
    """Whether the call goes to the prefill GEMM."""
    return route_for(m, n, k, **how) == "hk"


def route_decode(m: int, n: int, k: int, **how) -> bool:
    """Whether the call goes to `ops.decode_gemm`."""
    return route_for(m, n, k, **how) == "hk_decode"


def _arch_ok(device) -> bool:
    """Whether `device` is the arch the kernels are compiled for."""
    idx = device.index if device.index is not None else 0
    ok = _ARCH_OK.get(idx)
    if ok is None:
        import torch  # noqa: PLC0415

        from ..target import current_arch  # noqa: PLC0415

        name = torch.cuda.get_device_properties(idx).gcnArchName or ""
        ok = _ARCH_OK[idx] = name.split(":")[0] == current_arch()
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
    """y = x @ weight.T (+ bias), on the kernel the table names."""
    import torch  # noqa: PLC0415

    k = x.shape[-1]
    m = x.numel() // k if k else 0
    n = weight.shape[0]
    path = route_for(m, n, k,
                     bf16=x.dtype == torch.bfloat16 and weight.dtype == torch.bfloat16,
                     bias=bias is not None, contiguous_weight=weight.is_contiguous())
    if path == "fallback" or not _arch_ok(x.device):
        PATCH.count("fallback")
        return _fallback(x, weight, bias)
    x2 = x.reshape(m, k)
    if not x2.is_contiguous():
        x2 = x2.contiguous()
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    if path == "hk":
        _op_for(m, n, k)(x2, weight, out)
    else:
        from ..ops import decode_gemm  # noqa: PLC0415

        s = decode_gemm.schedule_for(m, k)
        if s["splits"] > 1:
            _decode_op(s)(x2, weight, out,
                          *decode_gemm.scratch_for(x.device, s["splits"], s["tokens"], n))
        else:
            _decode_op(s)(x2, weight, out)
    PATCH.count(path)
    return out.view(*x.shape[:-1], n)


def register_op():
    """Define `torch.ops.hk_vllm.linear`. Idempotent."""
    PATCH.define(linear)


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
    table = routes()
    if table is None:
        from ..target import current_arch  # noqa: PLC0415

        return (f"linear: not patched -- no routing table for {current_arch()}; "
                f"tools/hk-bench/route_tune.py writes one")
    register_op()
    _ORIG_DISPATCH = vu.dispatch_unquantized_gemm
    names = _common.rebind(PREFIXES, _ORIG_DISPATCH, dispatch_unquantized_gemm)
    return (f"linear -> hk where {table.source} measured it faster, on "
            f"{len(list(table.shapes()))} shapes ({len(names)} binding"
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
