"""Everything is in pyproject.toml; this only makes the wheel per-interpreter.

hk._mlir is the MLIR Python bindings built for one CPython (nanobind
extension modules), so a wheel carrying it is cp3XX-cp3XX-<platform>, not
py3-none-any. setuptools tags a wheel that way only for a distribution with
extension modules, and the bindings are built by CMake, not by setuptools --
hence the declaration. Build with scripts/build_wheel.sh, which stages the
headers and checks that hk._mlir matches the interpreter.
"""

from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
