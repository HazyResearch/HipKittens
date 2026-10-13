"""The module boundary around a translated kernel: pybind, torch, or bare."""

from . import scaffold
from .scaffold import render

__all__ = ["scaffold", "render"]
