"""hk -- HipKittens as a Python kernel IR for Radeon GPUs.

Write a tile kernel in Python, get a compiled HIP extension:

    import hk
    from hk import bf16

    @hk.kernel(arch="gfx1100",
               grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), p.o.batch))
    def add(a: hk.GL[bf16], b: hk.GL[bf16], o: hk.GL[bf16], *, ROWS=16, COLS=64):
        t   = hk.rt(bf16, ROWS, COLS)
        idx = hk.tile_coord(hk.block_idx.z, 0, hk.block_idx.y, hk.block_idx.x)
        hk.store(o, hk.load(a, idx, t) + hk.load(b, idx, t), idx)

    add(a, b, o)          # traces, generates C++, compiles with hipcc, launches

The backend is the stock hipcc compiling generated HipKittens C++ against
include/rdna3 -- not a custom LLVM. That is what lets a kernel written here run
inside any ROCm container, which is the whole point of shipping into vLLM and
SGLang.

Three things this layer exists to make impossible rather than merely
discouraged, each of which is a silent wrong answer on gfx11 and cost days to
find by hand:

  1. A spilling kernel is never returned. `s_waitcnt` in include/rdna3 is placed
     by hand; scratch traffic reorders against it and the kernel computes the
     wrong result at full speed with no diagnostic. hk.compile() raises.
  2. `s_waitcnt` carries no register dependence, so the compiler will hoist a
     WMMA above a wait it does not know is an ordering constraint. The emitter
     binds fragments after every wait; a human forgets.
  3. Only row-layout 16-bit operands reach ds_read_b128; a column operand
     degrades to 16 scalar ds_read_u16. The verifier says so at trace time
     rather than letting you find it in a benchmark.

Three entry points, in increasing order of what they need:
  `k.trace()`   pure Python -- no compiler, no GPU
  `k.build()`   hipcc, no GPU (this is where the gates run)
  `k(*tensors)` a GPU
"""

from . import autotune, codegen, ir, lang, ops, runtime, target
from .ir.nodes import DTYPES, DType, bf16, fp8e4m3, fp8e5m2, fp16, fp32, i8, i32
from .ir.verify import VerifyError
from .lang import *  # noqa: F401,F403
from .lang import __all__ as _lang_all
from .ops import attention
from .runtime import Build, CompileError, ResourceError
from .runtime.torch_ext import torch_op
from .runtime.arch import detect as detect_arch
from .target import TARGETS, get_target

__version__ = "0.1.0"

__all__ = [
    *_lang_all,
    "bf16", "fp16", "fp32", "i8", "i32", "fp8e4m3", "fp8e5m2", "DType", "DTYPES",
    "VerifyError", "CompileError", "ResourceError", "Build",
    "get_target", "TARGETS", "detect_arch",
    "autotune", "codegen", "ir", "lang", "ops", "runtime", "target",
    "torch_op",
    # The one op promoted out of `hk.ops`: it is the drop-in a
    # framework reaches for, and `hk.attention` is what the SDPA
    # patches in `kernels/rdna3/attn` already call it.
    "attention",
    "__version__",
]
