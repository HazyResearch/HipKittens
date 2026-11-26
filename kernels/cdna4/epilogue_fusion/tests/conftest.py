"""pytest loads this before the tests: puts kernels/cdna4/epilogue_fusion on sys.path so tests can `import hk`.
Shared test code is in helpers.py."""
import pathlib
import sys

EPI = str(pathlib.Path(__file__).resolve().parents[1])
if EPI not in sys.path:
    sys.path.insert(0, EPI)
