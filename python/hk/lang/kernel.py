"""The @hk.kernel decorator.

Parameter rules, which are the whole user-facing contract:

* Positional parameters are tensors. Annotate them `hk.GL[dtype]`.
* Keyword-only parameters are **constexpr**: they are baked into the generated
  source, so they are part of the compilation cache key, and inside the kernel
  body they are ordinary Python ints. That is deliberate -- a tile extent has to
  be a compile-time constant in C++, and making them real Python values means
  `range(D // 16)` and `if KV_BLOCK > 64` work without any tracing machinery.

Specialising on a constexpr produces a different kernel, and a different
compilation. `k.specialize(D=64)` is explicit about that; calling with a
different value does the same thing implicitly.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Tuple

from ..ir import builder
from ..ir.nodes import DType, GlobalType, KernelIR, Param, ScalarType, Value, fp32, i32
from . import host


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


class GL:
    """Annotation for a global tensor parameter: `a: hk.GL[bf16]`."""

    def __class_getitem__(cls, dtype: DType) -> GlobalType:
        if not isinstance(dtype, DType):
            raise TypeError(f"hk.GL[...] takes a dtype, got {dtype!r}")
        return GlobalType(dtype)


#: Annotation marker for a constexpr parameter. Optional -- every keyword-only
#: parameter is constexpr whether or not it is annotated -- but it documents the
#: intent at the call site, which matters when a kernel has a dozen knobs.
const = "hk.const"


class Kernel:
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
        #: importing hk.ops builds no IR and probes no GPU.
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
        self._ir_cache: Dict[Tuple, KernelIR] = {}
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

        A module with `from __future__ import annotations` -- which is most of
        them, and all of this package -- hands us the annotation as a string.
        Evaluating it in the defining module's namespace is what any typing
        library does, and skipping it would make the decorator quietly reject
        every kernel in such a file.
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
                f"in the module that defines the kernel, e.g. hk.GL[hk.bf16]."
            ) from None

    def _split_params(self):
        tensors, consts = [], {}
        for p in self.signature.parameters.values():
            if p.kind is inspect.Parameter.KEYWORD_ONLY:
                if p.default is inspect.Parameter.empty:
                    raise TypeError(
                        f"{self.name}: constexpr parameter {p.name!r} needs a default. "
                        f"Keyword-only parameters are baked into the generated source, "
                        f"so every one of them must have a value to bake."
                    )
                consts[p.name] = p.default
                continue
            if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                raise TypeError(f"{self.name}: *args / **kwargs are not supported")
            ann = self._annotation(p)
            if not isinstance(ann, GlobalType):
                raise TypeError(
                    f"{self.name}: parameter {p.name!r} must be annotated hk.GL[dtype] "
                    f"(got {ann!r}). Positional parameters are tensors; put scalars in "
                    f"keyword-only constexpr parameters."
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

    def trace(self, **const_overrides: Any) -> KernelIR:
        """Run the body once and return the IR. No GPU, no compiler."""
        consts = self._resolve_consts(const_overrides)
        key = self._key(consts)
        if key in self._ir_cache:
            return self._ir_cache[key]

        ir = KernelIR(name=self.name, arch=self.arch, warps=self.warps, consts=dict(consts))
        args = {}
        for pname, ptype in self._tensors:
            v = Value(ptype, name=pname)
            args[pname] = v
            ir.params.append(Param(pname, ptype))
        for cname, cval in consts.items():
            ir.params.append(
                Param(cname, ScalarType(_const_dtype(cval)), is_const=True, const_value=cval)
            )

        b = builder.Builder(ir)
        with builder.tracing(b):
            self.fn(**args, **consts)

        ir.grid = self._trace_grid(consts)

        from ..ir import passes

        passes.run_all(ir)
        self._ir_cache[key] = ir
        return ir

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
        from ..codegen import scaffold as sc

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
            from ..runtime import build as rt_build

            ir = self.trace(**const_overrides)
            self._build_cache[key] = rt_build(
                self.source(**const_overrides),
                ir.arch,
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
            from ..runtime import module as rtm

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


def kernel(
    fn: Optional[Callable] = None,
    *,
    arch: Optional[str] = None,
    warps: int = 1,
    grid: Optional[Callable] = None,
    name: Optional[str] = None,
    max_vgprs: Optional[int] = None,
    min_occupancy: Optional[int] = None,
):
    """Mark a function as a tile kernel. See the module docstring for the
    parameter rules."""

    def wrap(f):
        return Kernel(
            f, arch=arch, warps=warps, grid=grid, name=name,
            max_vgprs=max_vgprs, min_occupancy=min_occupancy,
        )

    return wrap(fn) if fn is not None else wrap
