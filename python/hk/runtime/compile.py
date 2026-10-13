"""hipcc, with a content-addressed cache and a spill gate.

A hipcc invocation for a real kernel costs 20-40 s, which is too slow to pay on
every import and far too slow to pay inside an autotune sweep. So the compiled
.so is keyed by everything that can change its bytes:

  * the generated source (which already folds in the kernel's Python source,
    its constexpr arguments and the target, since all three shaped the text),
  * the toolchain -- hipcc's own version string and the flags,
  * the contents of include/, because editing a header changes the kernel while
    leaving the generator's output byte-identical. This is the one people get
    wrong, and it is the one that produces a stale kernel that looks right.

Cache entries are never invalidated, only added: a key that hashes everything
does not need eviction for correctness, and `HK_CACHE_DIR` is a directory the
user can delete.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from ..target import Target, get_target
from . import resources
from .resources import KernelResources, ResourceError


class CompileError(Exception):
    pass


# ---------------------------------------------------------------- paths

#: Repo root, i.e. the directory that has include/. Works from a source
#: checkout (python/hk/runtime -> ../../..) and from an installed wheel, where
#: the headers are packaged under hk/include.
def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for cand in (here.parents[3], here.parents[1]):
        if (cand / "include" / "kittens.cuh").exists():
            return cand
    raise CompileError(
        f"cannot find include/kittens.cuh above {here}. hk needs the HipKittens "
        f"headers; set HK_INCLUDE_DIR to the directory containing them."
    )


def include_dir() -> Path:
    env = os.environ.get("HK_INCLUDE_DIR")
    if env:
        return Path(env)
    return _repo_root() / "include"


def cache_dir() -> Path:
    d = Path(
        os.environ.get("HK_CACHE_DIR")
        or Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "hk"
    )
    d.mkdir(parents=True, exist_ok=True)
    return d


def hipcc() -> str:
    for cand in (
        os.environ.get("HK_HIPCC"),
        shutil.which("hipcc"),
        "/opt/rocm/bin/hipcc",
    ):
        if cand and os.path.exists(cand):
            return cand
    raise CompileError(
        "no hipcc found. Set HK_HIPCC, or put ROCm's bin on PATH. hk compiles "
        "with the stock hipcc on purpose -- it needs no custom toolchain, which "
        "is what lets a generated kernel run in any ROCm container."
    )


# ---------------------------------------------------------------- flags


def _pybind_includes() -> List[str]:
    flags = [f"-I{sysconfig.get_path('include')}"]
    try:
        import pybind11  # noqa: PLC0415

        flags.append(f"-I{pybind11.get_include()}")
    except ImportError:
        # torch vendors the same headers; on a pod that has torch but not a
        # standalone pybind11 this is the path that works.
        try:
            import torch  # noqa: PLC0415

            flags.append(f"-I{Path(torch.__file__).parent / 'include'}")
        except ImportError:
            raise CompileError(
                "neither pybind11 nor torch is importable, so there are no "
                "pybind11 headers to compile against. pip install pybind11."
            ) from None
    return flags


def ext_suffix() -> str:
    return sysconfig.get_config_var("EXT_SUFFIX") or ".so"


def rocm_path() -> Path:
    env = os.environ.get("ROCM_PATH")
    if env:
        return Path(env)
    # hipcc lives at <rocm>/bin/hipcc or <rocm>/lib/llvm/bin; walk up to the
    # directory that has include/hip.
    for parent in Path(hipcc()).resolve().parents:
        if (parent / "include" / "hip").is_dir():
            return parent
    return Path("/opt/rocm")


def flags(target: Target, module_name: str, extra: Sequence[str] = ()) -> List[str]:
    """The hipcc command line, mirroring kernels/common.mk BUILD_MODE=pyext.

    include/hip is on the path explicitly because base_types.cuh includes
    <hip_bf16.h> unqualified; hipcc's own default search does not cover it.
    """
    return [
        f"-D{target.kittens_define}",
        f"--offload-arch={target.arch}",
        "-std=c++20",
        "-O3",
        "-w",
        f"-I{include_dir()}",
        f"-I{rocm_path() / 'include' / 'hip'}",
        *_pybind_includes(),
        f"-DHK_MODULE_NAME={module_name}",
        "-shared",
        "-fPIC",
        # The gate depends on this flag; see runtime/resources.
        "-Rpass-analysis=kernel-resource-usage",
        *extra,
    ]


# ---------------------------------------------------------------- key


def _tree_hash(root: Path, suffixes=(".cuh", ".hpp", ".h")) -> str:
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix in suffixes:
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


_toolchain_cache: dict = {}


def toolchain_fingerprint() -> str:
    """hipcc version + the include tree. Cached per process; hashing include/
    is ~10 ms and the answer cannot change mid-run."""
    exe = hipcc()
    if exe in _toolchain_cache:
        return _toolchain_cache[exe]
    try:
        ver = subprocess.run(
            [exe, "--version"], capture_output=True, text=True, timeout=120
        ).stdout
    except Exception as e:
        raise CompileError(f"{exe} --version failed: {e}") from e
    fp = hashlib.sha256(
        (ver + _tree_hash(include_dir())).encode()
    ).hexdigest()[:32]
    _toolchain_cache[exe] = fp
    return fp


#: Stand-in for the module name while keying. The real name is derived from the
#: key, so it cannot be an input to it; it also cannot change the generated code
#: in any way that matters, since it only names the pybind entry point.
_KEY_MODULE = "<module>"


_dialect_version: list = []


def dialect_version() -> str:
    """The hk dialect's semantic version, or "none" without hk._mlir. Not a
    hash of the library: a rebuild that changes nothing it means must not
    throw the compile cache away."""
    if not _dialect_version:
        try:
            from .._mlir.dialects import hk as hkd  # noqa: PLC0415

            _dialect_version.append(hkd.dialect_version())
        except Exception:  # noqa: BLE001
            _dialect_version.append("none")
    return _dialect_version[0]


def cache_key(source: str, target: Target, extra: Sequence[str] = ()) -> str:
    """Everything that can change the compiled bytes.

    The full flag list is in here, not just `extra`: hk's own flags change when
    hk changes, and a cache that ignores them serves the artifact from before
    the fix. That is exactly the failure mode the include-path bug would have
    hidden.
    """
    h = hashlib.sha256()
    h.update(toolchain_fingerprint().encode())
    h.update(dialect_version().encode())
    h.update("\0".join(flags(target, _KEY_MODULE, extra)).encode())
    h.update(source.encode())
    return h.hexdigest()[:32]


# ---------------------------------------------------------------- build


@dataclass
class Build:
    """One compiled artifact and everything used to judge it."""

    key: str
    module_name: str
    so_path: Path
    source_path: Path
    log_path: Path
    kernels: List[KernelResources]
    cached: bool

    def report(self) -> str:
        return resources.report(self.kernels)


def _load_cached(d: Path, target: Target) -> Optional[Build]:
    so = next(d.glob("*.so"), None) or next(d.glob("*.*.so"), None)
    marker = d / "OK"
    if so is None or not marker.exists():
        return None
    log = d / "build.log"
    try:
        ks = resources.parse(log.read_text(), target) if log.exists() else []
    except ResourceError:
        # A cache entry whose log we cannot read is still a valid .so -- the
        # gate passed when it was written, which is what the OK marker records.
        ks = []
    return Build(
        key=d.name,
        module_name=marker.read_text().strip(),
        so_path=so,
        source_path=d / "kernel.cpp",
        log_path=log,
        kernels=ks,
        cached=True,
    )


def build(
    source: str,
    arch: str,
    *,
    name: str = "hk_kernel",
    extra_flags: Sequence[str] = (),
    allow_scratch: bool = False,
    max_vgprs: Optional[int] = None,
    min_occupancy: Optional[int] = None,
    verbose: bool = False,
) -> Build:
    """Compile generated C++ into a loadable extension, or return the cached one.

    Raises ResourceError, and leaves nothing importable behind, if any
    instantiation spills.
    """
    target = get_target(arch)
    key = cache_key(source, target, extra_flags)
    out = cache_dir() / key

    if (hit := _load_cached(out, target)) is not None:
        # Re-gate on a hit. The key covers the source and the flags but not the
        # thresholds, so the same bytes can be asked for by one kernel with no
        # occupancy requirement and by another with a strict one; whoever
        # compiled first must not decide for the second. Cheap -- the numbers
        # come from the cached log, not from hipcc. Entries whose log no longer
        # parses gate on nothing, which is what the OK marker already attests.
        if hit.kernels:
            resources.gate(
                hit.kernels,
                allow_scratch=allow_scratch,
                max_vgprs=max_vgprs,
                min_occupancy=min_occupancy,
            )
        return hit

    module_name = f"hk_{name}_{key[:12]}"
    so_name = module_name + ext_suffix()

    # Build in a scratch directory and move into place, so a killed or failing
    # compile never leaves a half-written .so that the next run would import.
    tmp = Path(tempfile.mkdtemp(prefix="hk-build-", dir=cache_dir()))
    moved = False
    try:
        src = tmp / "kernel.cpp"
        src.write_text(source)
        cmd = [
            hipcc(), *flags(target, module_name, extra_flags),
            str(src), "-o", str(tmp / so_name),
        ]
        if verbose:
            print(" ".join(cmd), file=sys.stderr)

        proc = subprocess.run(cmd, capture_output=True, text=True)
        log = proc.stdout + proc.stderr
        (tmp / "build.log").write_text(log)
        (tmp / "command").write_text(" ".join(cmd))

        if proc.returncode != 0:
            raise CompileError(
                f"hipcc failed ({proc.returncode}) for kernel {name!r}.\n"
                f"Source kept at {_keep(tmp, key)}\n\n{log}"
            )

        kernels = resources.parse(log, target)
        try:
            resources.gate(
                kernels,
                allow_scratch=allow_scratch,
                max_vgprs=max_vgprs,
                min_occupancy=min_occupancy,
            )
        except ResourceError as e:
            raise ResourceError(
                f"{e}\n\nGenerated source kept at {_keep(tmp, key)}"
            ) from None

        (tmp / "OK").write_text(module_name)
        try:
            os.rename(tmp, out)
            moved = True
        except OSError:
            # A concurrent builder won the race. Its artifact is byte-identical
            # by construction -- the key hashes everything that could differ --
            # so losing is not an error.
            if not (out / "OK").exists():
                raise
            return _load_cached(out, target)
    finally:
        if not moved:
            shutil.rmtree(tmp, ignore_errors=True)

    return Build(
        key=key,
        module_name=module_name,
        so_path=out / so_name,
        source_path=out / "kernel.cpp",
        log_path=out / "build.log",
        kernels=kernels,
        cached=False,
    )


def _keep(tmp: Path, key: str) -> Path:
    """Preserve a failed build for inspection. A failure whose source is gone
    is much harder to act on than one that costs a few KB of disk."""
    dest = cache_dir() / f"failed-{key}"
    shutil.rmtree(dest, ignore_errors=True)
    shutil.copytree(tmp, dest)
    return dest
