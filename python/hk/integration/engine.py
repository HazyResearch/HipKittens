"""What every engine patch here has in common, written once.

A patch puts a generated kernel under a call the engine already makes. The
two so far -- vLLM's linear layers (`vllm_gemm`) and its attention backend
(`vllm_backend`) -- each wrote the same parts by hand:

* an opaque torch custom op with a fake implementation, so that under
  torch.compile the shape-dependent choice of kernel is not a guard and a
  recompile (`Op`);
* a route from shape to kernel, which was a rule transcribed from a
  measurement (`RouteTable`, which *is* the measurement);
* a fallback to what the engine would have run, and counters of which way
  calls went;
* a CUDA-graph contract, which was a comment beside a class attribute.

The contract is now checked. A patch declares `graph="safe"` -- every launch
is on torch's stream and depends on nothing but shapes, so a replay repeats
it -- or `graph="never"` with the reason, for a patch that plans on host
metadata a replay would not re-read. Under capture a "never" patch raises
`GraphCaptureError` with its reason instead of being frozen at the capture
batch's values, and a "safe" patch runs with torch's sync debug mode set to
error, so a host sync inside a wrapper -- `.item()`, `.tolist()`, a blocking
copy -- raises at the line that does it, naming the patch, instead of failing
the capture somewhere else or, worse, succeeding.
"""

from __future__ import annotations

import bisect
import contextlib
import functools
import json
import os
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

GRAPH_CONTRACTS = ("safe", "never")

#: Every patch constructed, by name, so a status line can report them all.
PATCHES: Dict[str, "Patch"] = {}


class GraphCaptureError(RuntimeError):
    """A patch did, under CUDA-graph capture, something a replay cannot repeat."""


def capturing() -> bool:
    """Whether the current stream is capturing a CUDA graph. No sync."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return False
    return torch.cuda.is_available() and torch.cuda.is_current_stream_capturing()


@contextlib.contextmanager
def no_host_sync(what: str):
    """Raise `GraphCaptureError` naming `what` at any host sync inside.

    torch's sync debug mode sees the syncs torch itself makes -- `.item()`,
    `.tolist()`, a blocking copy, `synchronize()` -- and not a raw HIP call
    inside an extension.
    """
    import warnings  # noqa: PLC0415

    import torch  # noqa: PLC0415

    prev = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Synchronization debug mode")
        torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    except RuntimeError as ex:
        if "synchroniz" not in str(ex) or isinstance(ex, GraphCaptureError):
            raise
        raise GraphCaptureError(
            f"{what} synchronized with the host while a CUDA graph was being "
            f"captured. A replay runs no Python, so whatever the wrapper read "
            f"back is frozen at the capture batch's value. Keep the decision "
            f"on host values the engine already has, or declare the patch "
            f"graph='never'. ({str(ex).splitlines()[0]})") from ex
    finally:
        torch.cuda.set_sync_debug_mode(prev)


class Patch:
    """One place an engine runs a generated kernel: its paths, counted, and
    its CUDA-graph contract, enforced by `guard`.

    The counters are Python, so a CUDA-graph replay -- which runs no Python --
    is not counted. Calls made while capturing are counted apart, in
    `captured`, so a table of where calls went cannot mix the two.
    """

    def __init__(self, name: str, paths: Sequence[str], *, graph: str = "safe",
                 why: str = ""):
        if graph not in GRAPH_CONTRACTS:
            raise ValueError(f"graph contract {graph!r} is not one of {GRAPH_CONTRACTS}")
        if graph != "safe" and not why:
            raise ValueError(f"{name}: a patch that cannot be captured says why")
        self.name = name
        self.paths = tuple(paths)
        self.graph = graph
        self.why = why
        self.stats: Dict[str, int] = {p: 0 for p in self.paths}
        self.captured: Dict[str, int] = {p: 0 for p in self.paths}
        PATCHES[name] = self

    def count(self, path: str) -> None:
        (self.captured if capturing() else self.stats)[path] += 1

    def reset(self) -> None:
        for d in (self.stats, self.captured):
            for p in d:
                d[p] = 0

    def check_capture(self) -> None:
        """Raise if this patch is being captured and has said it must not be."""
        if self.graph == "never" and capturing():
            raise GraphCaptureError(
                f"{self.name} is declared graph='never' and is being captured: "
                f"{self.why}")

    def guard(self, fn: Callable) -> Callable:
        """`fn` under this patch's contract. Free outside capture."""

        @functools.wraps(fn)
        def guarded(*args, **kwargs):
            if not capturing():
                return fn(*args, **kwargs)
            self.check_capture()
            with no_host_sync(self.name):
                return fn(*args, **kwargs)

        return guarded

    def summary(self) -> str:
        def fmt(d):
            return ", ".join(f"{p} {n}" for p, n in d.items())

        cap = sum(self.captured.values())
        return (f"{self.name}: {fmt(self.stats)}"
                + (f"; captured: {fmt(self.captured)}" if cap else ""))


class Op(Patch):
    """A patch that is an opaque torch custom op, `torch.ops.<ns>.<name>`.

    `schema` is the torch.library schema without the namespace; `fake` builds
    the output from shapes alone, which is what torch.compile traces.
    """

    def __init__(self, qualname: str, schema: str, fake: Callable,
                 paths: Sequence[str], **kw):
        super().__init__(qualname, paths, **kw)
        self.ns, self.op_name = qualname.split("::")
        self.schema = schema
        self.fake = fake
        self._lib = None

    @property
    def defined(self) -> bool:
        return self._lib is not None

    def define(self, impl: Callable, dispatch: str = "CUDA") -> "Op":
        """Register `impl`, guarded, and the fake. Idempotent."""
        if self._lib is not None:
            return self
        import torch  # noqa: PLC0415

        lib = torch.library.Library(self.ns, "FRAGMENT")
        lib.define(self.schema)
        lib.impl(self.op_name, self.guard(impl), dispatch)
        torch.library.register_fake(self.name, self.fake, lib=lib)
        self._lib = lib
        return self

    def __call__(self, *args, **kwargs):
        import torch  # noqa: PLC0415

        return getattr(getattr(torch.ops, self.ns), self.op_name)(*args, **kwargs)


class RouteTable:
    """Measured winners: for each (N, K), which path each measured M took.

    Written by tools/hk-bench/route_tune.py, never by hand. A shape's entry is
    a list of runs `[lo, hi, path]`: every measured M in [lo, hi] had `path`
    as its fastest by the tuner's margin. An M inside a run takes its path.
    An M between two runs lies between two measured Ms that disagreed, so it
    takes `fallback` -- the table does not guess which side of a tooth it is
    on. Above the largest measured M it takes the path measured there; below
    the smallest, and for a shape never measured, `fallback`.

    No buckets. GEMM time is a comb in M, not a curve, and a power-of-two
    bucket either routes the teeth or gives up the gaps between them.
    """

    def __init__(self, shapes: Dict[Tuple[int, int], dict], *, fallback: str = "fallback",
                 meta: Optional[dict] = None, source: str = ""):
        self.fallback = fallback
        self.meta = meta or {}
        self.source = source
        self._runs: Dict[Tuple[int, int], List[Tuple[int, int, str]]] = {}
        self._lows: Dict[Tuple[int, int], List[int]] = {}
        self._above: Dict[Tuple[int, int], str] = {}
        for nk, ent in shapes.items():
            rs = sorted((int(lo), int(hi), str(p)) for lo, hi, p in ent["runs"])
            for (_, hi, _), (lo2, _, _) in zip(rs, rs[1:]):
                if lo2 <= hi:
                    raise ValueError(f"{source}: runs for {nk} overlap at M={lo2}")
            self._runs[nk] = rs
            self._lows[nk] = [r[0] for r in rs]
            self._above[nk] = str(ent.get("above", rs[-1][2] if rs else fallback))
        #: (N, K) asked for and not in the table -- each is a shape a tuner
        #: run would route, and until then takes the engine's path.
        self.misses: set = set()

    @classmethod
    def from_json(cls, path) -> "RouteTable":
        path = Path(path)
        d = json.loads(path.read_text())
        shapes = {}
        for key, ent in d.get("shapes", {}).items():
            n, k = (int(v) for v in key.split("x"))
            shapes[(n, k)] = ent
        meta = {k: v for k, v in d.items() if k != "shapes"}
        return cls(shapes, meta=meta, source=str(path))

    def shapes(self) -> Iterable[Tuple[int, int]]:
        return self._runs.keys()

    def paths(self) -> set:
        return {p for rs in self._runs.values() for _, _, p in rs} | set(self._above.values())

    def lookup(self, m: int, n: int, k: int) -> str:
        rs = self._runs.get((n, k))
        if not rs:
            self.misses.add((n, k))
            return self.fallback
        i = bisect.bisect_right(self._lows[(n, k)], m) - 1
        if i < 0:
            return self.fallback
        lo, hi, path = rs[i]
        if m <= hi:
            return path
        return self._above[(n, k)] if i == len(rs) - 1 else self.fallback


def shipped_table(kind: str, arch: str, env: str) -> Optional[RouteTable]:
    """The routing table for `kind` on `arch`: `$env` if set, else the one in
    hk/tuned/, else None. A table measured on another arch is not loaded."""
    override = os.environ.get(env)
    path = Path(override) if override else (
        Path(__file__).resolve().parents[1] / "tuned" / f"routes_{kind}_{arch}.json")
    if not path.is_file():
        if override:
            raise FileNotFoundError(f"${env} names {path}, which does not exist")
        return None
    t = RouteTable.from_json(path)
    if t.meta.get("arch") not in (None, arch):
        raise ValueError(f"{path} was measured on {t.meta['arch']}, not {arch}")
    return t
