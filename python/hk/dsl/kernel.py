"""`@hk.kernel` and `@hk.func`: functions the DSL traces into hk modules.

Parameter rules, which are the whole user-facing contract:

* Positional parameters are tensors. Annotate them `hk.Tensor[dtype]`.
* Keyword-only parameters are **constexpr**: they are baked into the generated
  source, so they are part of the compilation cache key, and inside the kernel
  body they are ordinary Python values. That is deliberate -- a tile extent has
  to be a compile-time constant in C++, and making them real Python values
  means `range(D // 16)` unrolls and `if KIND == "softmax"` picks a branch at
  trace time.

Specialising on a constexpr produces a different kernel, and a different
compilation. `k.specialize(D=64)` is explicit about that; calling with a
different value does the same thing implicitly.

A trace is an MLIR module: the kernel `func.func` and its grid, verified and
run through the hk pass pipeline. `k.trace().dump()` prints it; `k.source()`
is the HipKittens C++ hk-translate writes from it.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

from .._mlir import ir
from .._mlir.dialects import func as func_d
from ..compiler import mlir as M
from ..errors import VerifyError
from ..types import DType, GlobalType, Param, ScalarType, fp32, i32
from . import builder as B
from . import host
from .module import begin_kernel, build_grid
from .rewrite import rewrite
from .values import Global


def _const_dtype(value: Any) -> DType:
    """Constexpr params are ints in every kernel we have; floats (a softmax
    scale, say) are allowed and are baked in as literals just the same."""
    if isinstance(value, bool) or isinstance(value, int):
        return i32
    if isinstance(value, float):
        return fp32
    raise TypeError(
        f"constexpr parameter value {value!r} is a {type(value).__name__}; "
        f"only ints and floats can be baked into the generated source"
    )


class Tensor:
    """The annotation for a tensor parameter: `x: hk.Tensor[hk.bf16]`."""

    def __class_getitem__(cls, dtype: DType) -> GlobalType:
        if not isinstance(dtype, DType):
            raise TypeError(f"hk.Tensor[...] takes a dtype, got {dtype!r}")
        return GlobalType(dtype)


class Constexpr:
    """Marks a constexpr parameter, with its type if you like:
    `EPS: hk.Constexpr[float] = 1e-6`. Every keyword-only parameter is a
    constexpr whether or not it says so -- the marker documents it, which
    matters when a kernel has a dozen knobs -- and a positional one is a
    constexpr when it is annotated so."""

    def __class_getitem__(cls, _type):
        return cls


def _loc_text(loc) -> str:
    """"file:line in func" out of a location the frontend built."""
    try:
        if loc.is_a_name():
            loc = loc.child_loc
        if loc.is_a_fused():
            meta = loc.fused_metadata if hasattr(loc, "fused_metadata") else None
            locs = loc.locations
            if locs and locs[0].is_a_file():
                f = locs[0]
                fn = ir.StringAttr(meta).value if meta is not None else "?"
                return f"{f.filename}:{f.start_line} in {fn}"
    except Exception:  # noqa: BLE001 -- best effort, never mask the error
        pass
    return ""


def verify_error(e: Exception) -> VerifyError:
    """An MLIRError as the VerifyError a kernel author reads."""
    diags = getattr(e, "error_diagnostics", None) or []
    if not diags:
        return VerifyError(str(e))
    d = diags[0]
    msg = str(d.message)
    if "\n    at " not in msg:
        at = _loc_text(d.location)
        if at:
            msg += f"\n    at {at}"
    return VerifyError(msg)


class TracedKernel:
    """One specialization of a kernel, traced and lowered: its hk module and
    what the module boundary needs to know about it."""

    def __init__(self, name: str, arch: str, warps: int, consts: Dict[str, Any],
                 tensors: List, module: ir.Module):
        self.name = name
        self.arch = arch
        self.warps = warps
        self.consts = consts
        self.module = module
        self._tensors = [Param(n, t) for n, t in tensors]
        self.params = self._tensors + [
            Param(k, ScalarType(_const_dtype(v)), is_const=True, const_value=v)
            for k, v in consts.items()]
        k = M.kernel_of(module)
        with M.context():
            self.lds_bytes = ir.IntegerAttr(k.attributes["hk.lds_bytes"]).value
            self.warnings = [ir.StringAttr(a).value
                             for a in ir.ArrayAttr(k.attributes["hk.warnings"])]

    @property
    def threads(self) -> int:
        return self.warps * 32

    def tensors(self) -> List[Param]:
        return list(self._tensors)

    def written_tensors(self) -> List[str]:
        """The tensors the kernel stores into, in declaration order."""
        k = M.kernel_of(self.module)
        names = [p.name for p in self._tensors]
        args = list(k.regions[0].blocks[0].arguments)
        out = set()

        def visit(op):
            if op.name in ("hk.store", "hk.store_scalar", "hk.group_store"):
                dst = op.operands[0]
                for i, a in enumerate(args):
                    if a == dst:
                        out.add(names[i])
            for r in op.regions:
                for b in r.blocks:
                    for o in b.operations:
                        visit(o)

        with M.context():
            for o in k.regions[0].blocks[0].operations:
                visit(o)
        return [n for n in names if n in out]

    def cpp(self, includes) -> str:
        return M.translate(self.module, includes)

    def dump(self) -> str:
        """The module as MLIR, values named as the C++ names them: `%ld1`
        here is `ld1` in `cpp()`. Printed from a copy, renamed by the
        translation's own naming, so the two cannot disagree."""
        from .._mlir.dialects import hk as hkd  # noqa: PLC0415

        with M.context():
            mod = self.module.operation.clone()
            for op in mod.regions[0].blocks[0].operations:
                if "hk.kernel" in op.attributes:
                    hkd.name_values_as_cpp(op)
            text = mod.get_asm(use_name_loc_as_prefix=True)
            mod.erase()
            return text

    def __repr__(self) -> str:
        return f"<hk.TracedKernel {self.name} arch={self.arch} warps={self.warps}>"


class Kernel:
    """A tile kernel. Call it to launch; `trace`, `source` and `build` stop
    earlier, and need no GPU."""

    def __init__(
        self,
        fn: Callable,
        *,
        arch: Optional[str] = None,
        warps: int = 1,
        grid: Optional[Callable] = None,
        name: Optional[str] = None,
        max_vgprs: Optional[int] = None,
        min_occupancy: Optional[int] = None,
    ):
        self.fn = fn
        #: None means "whatever this process builds for" -- see
        #: hk.target.current_arch. Resolved on first read, not here, so that
        #: importing hk.ops traces nothing and probes no GPU.
        self._arch = arch
        self.warps = warps
        self.grid_fn = grid
        self.name = name or fn.__name__
        #: Extra build gates, on top of the unconditional no-spill rule. Set
        #: these when a kernel is only worth having at a given occupancy, so a
        #: change that quietly costs a wave fails the build instead of the
        #: benchmark.
        self.max_vgprs = max_vgprs
        self.min_occupancy = min_occupancy
        self.signature = inspect.signature(fn)
        self._tensors, self._defaults = self._split_params()
        self._ir_cache: Dict[Tuple, TracedKernel] = {}
        self._build_cache: Dict[Tuple, Any] = {}
        self._mod_cache: Dict[Tuple, Any] = {}
        self._spec_cache: Dict[Tuple, "Kernel"] = {}
        #: constexpr kwargs, exactly as a caller passed them, -> the bound
        #: pybind launcher. See __call__ for why this is keyed on the raw
        #: kwargs rather than on the resolved constexprs.
        self._launch_cache: Dict[Tuple, Any] = {}
        functools.update_wrapper(self, fn)

    @property
    def arch(self) -> str:
        if self._arch is None:
            from ..target import current_arch  # noqa: PLC0415

            return current_arch()
        return self._arch

    # -- signature ---------------------------------------------------------

    def _annotation(self, p: inspect.Parameter):
        """The parameter's annotation as a value.

        A module with `from __future__ import annotations` hands us the
        annotation as a string. Evaluating it in the defining module's
        namespace is what any typing library does, and skipping it would make
        the decorator quietly reject every kernel in such a file.
        """
        ann = p.annotation
        if not isinstance(ann, str):
            return ann
        g = getattr(self.fn, "__globals__", {})
        # A kernel built by a factory names the factory's arguments in its
        # annotations -- `x: hk.Tensor[dtype]` -- so the closure is in scope.
        code = getattr(self.fn, "__code__", None)
        cells = getattr(self.fn, "__closure__", None) or ()
        local = {}
        for n, c in zip(code.co_freevars if code else (), cells):
            try:
                local[n] = c.cell_contents
            except ValueError:  # an empty cell
                pass
        try:
            return eval(ann, g, local)  # noqa: S307 -- the module's own source
        except Exception as e:
            raise TypeError(
                f"{self.name}: cannot resolve the annotation {ann!r} on "
                f"parameter {p.name!r}: {e}. It must name something importable "
                f"in the module that defines the kernel, e.g. hk.Tensor[hk.bf16]."
            ) from None

    def _split_params(self):
        tensors, consts = [], {}
        for p in self.signature.parameters.values():
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError(f"{self.name}: *args / **kwargs are not supported")
            keyword_only = p.kind is inspect.Parameter.KEYWORD_ONLY
            ann = None if keyword_only else self._annotation(p)
            if keyword_only or ann is Constexpr:
                if p.default is inspect.Parameter.empty:
                    raise TypeError(
                        f"{self.name}: constexpr parameter {p.name!r} needs a default. "
                        f"Constexpr parameters are baked into the generated source, "
                        f"so every one of them must have a value to bake."
                    )
                consts[p.name] = p.default
                continue
            if not isinstance(ann, GlobalType):
                raise TypeError(
                    f"{self.name}: parameter {p.name!r} must be annotated "
                    f"hk.Tensor[dtype] (got {ann!r}). Positional parameters are "
                    f"tensors; a scalar is a constexpr -- keyword-only, or "
                    f"annotated hk.Constexpr[int] with a default."
                )
            tensors.append((p.name, ann))
        return tensors, consts

    def _resolve_consts(self, overrides: Dict[str, Any]) -> Dict[str, Any]:
        unknown = set(overrides) - set(self._defaults)
        if unknown:
            raise TypeError(
                f"{self.name}: no constexpr parameter(s) {sorted(unknown)}; "
                f"known: {sorted(self._defaults)}"
            )
        return {**self._defaults, **overrides}

    @staticmethod
    def _key(consts: Dict[str, Any]) -> Tuple:
        return tuple(sorted(consts.items()))

    # -- tracing -----------------------------------------------------------

    def trace(self, **const_overrides: Any) -> TracedKernel:
        """Run the body once into an hk module and run the passes over it.
        No GPU, no compiler."""
        consts = self._resolve_consts(const_overrides)
        key = self._key(consts)
        got = self._ir_cache.get(key)
        if got is not None:
            return got
        arch = self.arch
        t = B.Trace(self.name, arch)
        body = rewrite(self.fn)
        with B.tracing(t), ir.Location.unknown():
            mod, f, entry = begin_kernel(self.name, arch, self.warps, consts,
                                         self._tensors)
            t.module, t.func, t.entry = mod, f, entry
            args = {n: Global(a, ty, n)
                    for (n, ty), a in zip(self._tensors, entry.arguments)}
            with t.at(ir.InsertionPoint(entry)):
                body(**args, **consts)
                func_d.ReturnOp([], loc=ir.Location.unknown())
            build_grid(mod, self.name, self._tensors, self._trace_grid(consts))
        try:
            with M.context():
                mod.operation.verify()
            M.run_pipeline(mod)
        except ir.MLIRError as e:
            raise verify_error(e) from None
        traced = self._ir_cache[key] = TracedKernel(
            self.name, arch, self.warps, dict(consts), list(self._tensors), mod)
        return traced

    def _trace_grid(self, consts: Dict[str, Any]):
        if self.grid_fn is None:
            raise TypeError(
                f"{self.name}: no grid. Pass grid=lambda p: (x, y, z) to @hk.kernel; "
                f"p.<tensor>.<batch|depth|rows|cols> gives an extent and "
                f"p.<CONSTEXPR> gives a constexpr value."
            )
        ref = host.ParamRef([n for n, _ in self._tensors], consts)
        dims = self.grid_fn(ref)
        if not isinstance(dims, tuple) or len(dims) != 3:
            raise TypeError(
                f"{self.name}: grid lambda must return a 3-tuple (x, y, z), got {dims!r}"
            )
        return dims

    # -- compilation / launch ---------------------------------------------

    def source(self, scaffold: str = "pybind", **const_overrides: Any) -> str:
        """The generated HipKittens C++. Readable on purpose -- it is the
        artifact you take to /kernel-resource-check when something spills."""
        from ..codegen import scaffold as sc  # noqa: PLC0415

        return sc.render(self.trace(**const_overrides), scaffold)

    def build(self, **const_overrides: Any):
        """Generate and compile. Returns a runtime.Build, which carries the
        per-instantiation resource numbers as well as the .so.

        The compile step gates on ScratchSize and register spills: a kernel that
        spills is not slow here, it is *wrong*, because include/rdna3 hand-manages
        s_waitcnt. A spilling build raises rather than being returned.

        Needs hipcc but not a GPU, which is what makes the whole development
        loop runnable on a machine with no Radeon in it.
        """
        consts = self._resolve_consts(const_overrides)
        key = self._key(consts)
        if key not in self._build_cache:
            from ..runtime import build as rt_build  # noqa: PLC0415

            traced = self.trace(**const_overrides)
            self._build_cache[key] = rt_build(
                self.source(**const_overrides),
                traced.arch,
                name=self.name,
                max_vgprs=self.max_vgprs,
                min_occupancy=self.min_occupancy,
            )
        return self._build_cache[key]

    def compile(self, **const_overrides: Any):
        """Generate, compile and load. Returns the loaded extension module."""
        consts = self._resolve_consts(const_overrides)
        key = self._key(consts)
        if key not in self._mod_cache:
            from ..runtime import module as rtm  # noqa: PLC0415

            b = self.build(**const_overrides)
            self._mod_cache[key] = rtm.load(b.so_path, b.module_name)
        return self._mod_cache[key]

    def __call__(self, *tensors: Any, **const_overrides: Any):
        """Launch. Everything before the launch is a dict lookup.

        This is on the critical path of every op in `hk.ops`, and at these
        sizes the critical path is short: a 4096x4096 quantize is 60 us of
        GPU, so ten microseconds of Python in front of it is sixteen percent
        and the difference between beating torch.compile and losing to it.
        What used to be here -- resolve the constexprs, sort them into a key,
        look up the module, getattr the entry point -- built two sets, a
        merged dict and a sorted tuple per launch.

        So the launcher is memoized on the override kwargs *as passed*. That
        is a weaker key than the resolved constexprs: `k(x, o)` and
        `k(x, o, TPW=0)` are the same launch and get two entries. It is also a
        key that costs one tuple and one dict probe to compute, where the
        resolved one costs a dict merge and a sort, and the duplicate entries
        are bounded by the number of distinct call sites. The slow path still
        runs once per distinct kwargs tuple and still does every check.
        """
        if len(tensors) != len(self._tensors):
            names = [n for n, _ in self._tensors]
            raise TypeError(
                f"{self.name}() takes {len(names)} tensors {names}, got {len(tensors)}"
            )
        try:
            launch = self._launch_cache[tuple(const_overrides.items())]
        except (KeyError, TypeError):
            # TypeError: an unhashable constexpr value. It will be rejected
            # downstream with a better message than this would give.
            mod = self.compile(**const_overrides)
            launch = getattr(mod, self.name)
            try:
                self._launch_cache[tuple(const_overrides.items())] = launch
            except TypeError:
                pass
        return launch(*tensors)

    def specialize(self, **const_overrides: Any) -> "Kernel":
        """A view of this kernel with different constexpr defaults.

        Memoized, and that is not an optimization detail. A fresh Kernel has
        empty trace / build / module caches, so calling `specialize()` on every
        invocation -- which is what a wrapper like `rmsnorm(x, eps=...)`
        naturally does -- re-traces the body and re-resolves the extension
        module on every launch. Measured on the W7900D that was ~3.5 ms of
        Python in front of a 0.5 ms kernel: rmsnorm and layernorm came out
        flat in the tensor size, which is the signature of a fixed host cost
        rather than a slow kernel.

        The cheaper surface is to pass the constexpr to the call itself --
        `k(x, o, EPS=eps)` -- which uses the caches directly and makes no
        object at all. specialize() is for handing a configured kernel to
        something that will call it many times.
        """
        key = self._key(self._resolve_consts(const_overrides))
        if key not in self._spec_cache:
            k = type(self)(
                self.fn, arch=self._arch, warps=self.warps, grid=self.grid_fn,
                name=self.name, max_vgprs=self.max_vgprs,
                min_occupancy=self.min_occupancy,
            )
            k._defaults = dict(key)
            self._spec_cache[key] = k
        return self._spec_cache[key]

    def __repr__(self) -> str:
        return f"<hk.Kernel {self.name} arch={self.arch} warps={self.warps}>"


def kernel(fn=None, *, arch: Optional[str] = None, warps: int = 1, grid=None,
           name: Optional[str] = None, max_vgprs: Optional[int] = None,
           min_occupancy: Optional[int] = None):
    """Mark a function as a tile kernel. See the module docstring for the
    parameter rules."""

    def wrap(f):
        return Kernel(f, arch=arch, warps=warps, grid=grid, name=name,
                      max_vgprs=max_vgprs, min_occupancy=min_occupancy)

    return wrap(fn) if fn is not None else wrap


def func(fn):
    """A device helper: traced inline where it is called, with its `for`, `if`
    and `while` rewritten like a kernel's."""

    @functools.wraps(fn)
    def call(*args, **kwargs):
        if B.current_or_none() is None:
            raise RuntimeError(f"{fn.__name__} is an @hk.func; call it from a "
                               f"kernel being traced")
        return rewrite(fn)(*args, **kwargs)

    call.__hk_func__ = fn
    return call


__all__ = ["kernel", "func", "Kernel", "TracedKernel", "Tensor", "Constexpr",
           "verify_error"]
