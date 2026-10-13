"""Kernels written in the DSL and shipped with the package.

Importing this module builds the IR for nothing: kernels are traced on first
use and compiled on first launch. `import hk` therefore costs no hipcc.
"""

from . import attn, decode_gemm, elementwise, fused, gemm, norm, paged, quant, sdpa
from .attn import attention
from .elementwise import (
    abs,  # noqa: A004 -- torch.abs, not the builtin
    add,
    exp,
    gelu,
    maximum,
    minimum,
    mul,
    neg,
    relu,
    sub,
)
from .fused import rope, silu_mul
from .gemm import matmul
from .norm import layernorm, rmsnorm, softmax
from .quant import dequantize, quantize

__all__ = [
    "attn", "sdpa", "elementwise", "fused", "gemm", "decode_gemm", "norm", "quant",
    "add", "sub", "mul", "maximum", "minimum",
    "exp", "relu", "gelu", "neg", "abs",
    "rmsnorm", "layernorm", "softmax",
    "silu_mul", "rope",
    "quantize", "dequantize",
    "matmul",
    "attention",
]
