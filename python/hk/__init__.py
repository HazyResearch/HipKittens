"""hk -- HipKittens kernels for Radeon GPUs, written in Python.

    import hk

    @hk.kernel(grid=lambda p: (hk.cdiv(p.o.cols, 64), hk.cdiv(p.o.rows, 16), p.o.batch))
    def add(a: hk.Tensor[hk.bf16], b: hk.Tensor[hk.bf16], o: hk.Tensor[hk.bf16]):
        idx = hk.tile_coord(hk.arch.block_idx.z, 0, hk.arch.block_idx.y,
                            hk.arch.block_idx.x)
        t = hk.RT(hk.bf16, 16, 64)
        o[idx] = a[idx].load(t) + b[idx].load(t)

    add(a, b, o)    # traces, verifies, translates to C++, compiles, launches

A kernel is traced into the hk MLIR dialect (mlir/): tiles are buffers, scalars
are arith, a Python `for`/`if`/`while` on device values is scf. The passes --
verification, LDS allocation, the LDS wait pipeline, the register budget --
and the translation to HipKittens C++ are C++. The backend is the stock hipcc
compiling that C++ against include/rdna3, not a custom LLVM, which is what
lets a kernel written here run inside any ROCm container -- the point of
shipping into vLLM and SGLang.

Three things this layer exists to make impossible rather than merely
discouraged, each of which is a silent wrong answer on gfx11 and cost days to
find by hand:

  1. A spilling kernel is never returned. `s_waitcnt` in include/rdna3 is placed
     by hand; scratch traffic reorders against it and the kernel computes the
     wrong result at full speed with no diagnostic. `k.build()` raises.
  2. `s_waitcnt` carries no register dependence, so the compiler will hoist a
     WMMA above a wait it does not know is an ordering constraint. The
     translation binds fragments after every wait; a human forgets.
  3. Only row-layout 16-bit operands reach ds_read_b128; a column operand
     degrades to 16 scalar ds_read_u16. The passes say so at trace time
     rather than letting you find it in a benchmark.

Three entry points, in increasing order of what they need:
  `k.trace()`   the hk module -- no compiler, no GPU
  `k.build()`   hipcc, no GPU (this is where the gates run)
  `k(*tensors)` a GPU
"""

from . import autotune, codegen, compiler, dsl, ops, runtime, target
from .dsl import *  # noqa: F401,F403 -- the language is the package
from .dsl import __all__ as _dsl_all
from .errors import VerifyError
from .ops import attention
from .runtime import Build, CompileError, ResourceError
from .runtime.arch import detect as detect_arch
from .runtime.torch_ext import torch_op
from .target import TARGETS, get_target
from .types import DTYPES

__version__ = "0.2.0"

__all__ = [
    *_dsl_all,
    "DTYPES", "VerifyError", "CompileError", "ResourceError", "Build",
    "get_target", "TARGETS", "detect_arch",
    "autotune", "codegen", "compiler", "dsl", "ops", "runtime", "target",
    "torch_op",
    # The one op promoted out of `hk.ops`: it is the drop-in a
    # framework reaches for, and `hk.attention` is what the SDPA
    # patches in `kernels/rdna3/attn` already call it.
    "attention",
    "__version__",
]
