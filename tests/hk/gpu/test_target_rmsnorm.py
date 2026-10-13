"""The RMSNorm the MLIR plan set as the language's target, run against torch:
reading like the math is worth nothing if it does not compute it."""

from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ir"))
from test_target_syntax import rmsnorm_kernel  # noqa: E402

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("rows, cols", [(16, 256), (48, 1024), (32, 4096)])
def test_it_matches_torch(rows, cols):
    x = torch.randn(1, 1, rows, cols, device="cuda", dtype=torch.bfloat16)
    o = torch.empty_like(x)
    rmsnorm_kernel()(x, o)
    xf = x.float()
    want = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)
    torch.testing.assert_close(o.float(), want, rtol=2e-2, atol=2e-2)
