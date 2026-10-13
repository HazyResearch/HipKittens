"""Architecture facts, as data.

Every number here was measured on the part, not read off a datasheet. The
provenance of each one is in the module that defines it. One definition, shared
by the verifier, the LDS allocator, the register-budget model, the autotuner and
the C++ emitter -- these facts used to be scattered across RDNA.md, kernel
comments and a notebook, and they drifted.
"""

from .base import MmaShape, Target
from .gfx1100 import GFX1100
from .gfx1201 import GFX1201

TARGETS = {t.arch: t for t in (GFX1100, GFX1201)}

# Accept the marketing names too; the Makefiles use these.
_ALIASES = {
    "RDNA3": "gfx1100",
    "RDNA4": "gfx1201",
    "gfx11": "gfx1100",
    "gfx12": "gfx1201",
}


def get_target(arch: str):
    """Look up a target by gfx name or by the GPU_TARGET name common.mk uses."""
    key = _ALIASES.get(arch, arch)
    if key not in TARGETS:
        raise KeyError(
            f"unknown arch {arch!r}; known: {sorted(TARGETS)} "
            f"(aliases: {sorted(_ALIASES)}). Adding one means running the "
            f"probes in tools/rdna-probes, not editing a table."
        )
    return TARGETS[key]


_CURRENT = []


def current_arch() -> str:
    """The arch a kernel that names none is built for, decided once per process.

    hk.runtime.arch.detect's answer -- `$HK_ARCH` first, which is the only
    way to build for a part that is not in the machine and how the gfx1201
    kernels get their compile gate here; then the GPU torch or rocminfo sees.
    Else gfx1100, the part this library was measured on: tracing and
    compiling need no GPU, and a box without one should get the source a
    W7900 would.
    """
    if not _CURRENT:
        from ..runtime.arch import detect  # noqa: PLC0415

        arch = detect() or ""
        if _ALIASES.get(arch, arch) not in TARGETS:
            arch = "gfx1100"
        _CURRENT.append(_ALIASES.get(arch, arch))
    return _CURRENT[0]


__all__ = ["GFX1100", "GFX1201", "TARGETS", "current_arch", "get_target",
           "Target", "MmaShape"]
