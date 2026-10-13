"""Schedule search, with the resource gate in front of the GPU.

This is what replaces `kernels/rdna3/*/sweep.sh`. That script edits a `#define`,
runs `make clean`, recompiles, launches the benchmark, greps the number, and
repeats -- one candidate at a time, each in its own process, every one of them
reaching the GPU whether or not it had any business being there.

Three things are wrong with that, and this module is shaped by all three.

**Most candidates can be rejected without a GPU.** A schedule that spills is not
a slow schedule, it is a wrong one (include/rdna3 places `s_waitcnt` by hand and
scratch traffic reorders against it), and `Kernel.build()` already refuses to
return a `.so` for it. A schedule that drops an occupancy wave is visible in the
same compile. So stage 1 *compiles* the whole space and keeps only what passes
the gate; nothing that spills ever costs a GPU run, and on the GEMM space that
is most of the space.

**The compiles are independent.** hipcc is 4-40 s per kernel and a space is
dozens of them. Stage 1 hands the builds to `runtime.warm.warm_cache`, which
traces serially (the IR builder is a module-global stack) and compiles
`workers` at a time. Same code path as cache warming, because it is the same
problem.

**Cross-process A/B on this chip is invalid.** The W7900D's clock and power
state differs enough between process launches to manufacture a 13-16% gap out
of nothing, so a sweep that launches one process per candidate measures the
launch order as much as the schedule. Stage 2 times every survivor in *one*
process, in interleaved rounds whose order reverses, and keeps each candidate's
minimum. Drift that is monotone in wall-clock then hits both ends of the table
and falls out of the ranking.

The output is a record on disk: a schedule, the measurement that chose it, and
the toolchain fingerprint it was chosen under. Production reads the record and
compiles exactly one kernel. `python3 -m hk.autotune --aot` prebuilds every
recorded schedule into the content-addressed cache, which is what takes the JIT
off the serving path -- a `pip install`ed wheel plus a warmed cache launches
with a dict lookup in front of it.

    python3 -m hk.autotune --list              # what is tuned, and to what
    python3 -m hk.autotune --tune gemm_bf16    # run the search (needs a GPU)
    python3 -m hk.autotune --aot               # prebuild every recorded schedule
"""

from __future__ import annotations

import itertools
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..errors import VerifyError
from ..runtime.compile import cache_dir, toolchain_fingerprint
from ..runtime.resources import ResourceError

__all__ = [
    "space", "label_of", "Tuner", "TuneResult", "Row", "REGISTRY",
    "records_dir", "shipped_dir", "enabled", "load_records", "save_record",
    "forget_records", "generation", "main",
]


# ------------------------------------------------------------------ the space


def space(constraint: Optional[Callable[[Dict[str, Any]], bool]] = None,
          **axes: Sequence[Any]) -> List[Dict[str, Any]]:
    """The cartesian product of `axes`, in a stable order, optionally filtered.

    Deliberately a list and not a generator: a schedule space is a few dozen
    dicts, it goes into a record and into error messages, and `len()` of it is
    the first thing anyone asks. The order is the order the axes were given,
    first axis slowest, so a space reads the way it was written.

    `constraint` is for cheap arithmetic that keeps an obviously-dead corner out
    of the report. It is *not* where validation lives: a kernel factory that
    refuses a schedule raises, and a raise is recorded as an `invalid` row with
    its message rather than silently dropped.
    """
    names = list(axes)
    out = []
    for combo in itertools.product(*(list(axes[n]) for n in names)):
        s = dict(zip(names, combo))
        if constraint is None or constraint(s):
            out.append(s)
    return out


def label_of(schedule: Dict[str, Any]) -> str:
    """A short, filename-safe, order-stable name for one point in a space."""
    return "_".join(f"{k}{v}" for k, v in schedule.items())


# ------------------------------------------------------------------ results


@dataclass
class Row:
    """One candidate and what became of it.

    `status` is one of:
      `ok`        built, passed the gate, timed
      `invalid`   the kernel factory refused the schedule (bad warp grid, LDS
                  overflow) -- no compile was attempted
      `rejected`  compiled, but the gate refused it: it spills, or it misses
                  the occupancy floor
      `error`     the trace, the compile or the benchmark failed otherwise

    The failure kinds are kept apart because they mean different things to
    whoever reads the table. `invalid` is arithmetic and is free; `rejected` is
    the gate doing its job and is the interesting column -- it is the half of
    the output a sweep script throws away, and it is what says why the big
    tiling is not in the ranking; `error` is a bug and should not be in a space.
    """

    label: str
    schedule: Dict[str, Any]
    status: str
    ms: Optional[float] = None
    reason: str = ""
    vgpr: Optional[int] = None
    occupancy: Optional[int] = None
    scratch: Optional[int] = None
    times: List[float] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def line(self) -> str:
        ms = f"{self.ms:9.3f}" if self.ms is not None else f"{'-':>9}"
        res = f"  vgpr={self.vgpr} occ={self.occupancy}" if self.vgpr is not None else ""
        tail = f"  {self.reason}" if self.reason else ""
        return f"{self.label:<36}{self.status:<10}{ms}{res}{tail}"


@dataclass
class TuneResult:
    rows: List[Row] = field(default_factory=list)
    best: Optional[Row] = None
    seconds: float = 0.0
    key: str = ""

    @property
    def survivors(self) -> List[Row]:
        return [r for r in self.rows if r.ok]

    def report(self) -> str:
        head = f"{'schedule':<36}{'status':<10}{'ms':>9}"
        body = "\n".join(r.line() for r in sorted(
            self.rows, key=lambda r: (r.ms is None, r.ms or 0.0, r.label)))
        win = (f"best: {self.best.label} at {self.best.ms:.3f} ms"
               if self.best else "best: nothing survived the gate")
        return (f"{head}\n{body}\n{win}\n{len(self.survivors)}/{len(self.rows)} "
                f"candidates measured in {self.seconds:.1f}s")


# ------------------------------------------------------------------ records


def records_dir() -> Path:
    """Where tuning writes. Beside the compile cache, because the two go stale
    together: a new hipcc or a changed include/rdna3 moves the toolchain
    fingerprint, and a schedule measured under the old one is a guess."""
    d = Path(os.environ.get("HK_TUNE_DIR", cache_dir() / "autotune"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def shipped_dir() -> Path:
    """Records that travel with the package -- the AOT half.

    A wheel carries the schedules measured on the arch it names, so a fresh
    install picks the right tiling on its first call instead of the default
    one, and `--aot` can prebuild them before any request arrives. User records
    win over shipped ones: whoever ran the tuner on *this* machine measured
    this machine.
    """
    # hk/tuned, not hk/autotune/tuned: the records are package data of `hk`
    # (see pyproject's package-data), and they are read by anything that
    # resolves a schedule, not only by the tuner that wrote them.
    return Path(__file__).resolve().parent.parent / "tuned"


#: The smallest recorded win that is allowed to change what production
#: launches, as a fraction. Both `--ship` and the lookup path apply it.
#:
#: It exists because `min` over a column of noisy numbers always names a
#: winner. The attention sweep is the case that forced it: 32 schedules, six
#: of them legal, and the best beat the default by 0.10-0.21% on every key --
#: four different winners across three sizes, which is what a coin looks like.
#: Shipping that would have pinned four kernels on a coin flip.
#:
#: Applied at lookup and not only at ship time, so that a record measured on
#: *this* machine is held to the same bar as one that travelled in a wheel. A
#: record with no `gain` at all predates the field and is honoured: it is a
#: measurement we cannot judge, not one we judged and rejected.
MIN_GAIN = 0.01


def min_gain() -> float:
    try:
        return float(os.environ["HK_AUTOTUNE_MIN_GAIN"])
    except (KeyError, ValueError):
        return MIN_GAIN


def enabled() -> bool:
    """`HK_AUTOTUNE=0` pins every tuner to its default schedule.

    One switch, because the failure mode it exists for is "the tuned schedule
    was faster on the box it was tuned on and is not faster here", and the
    first thing to do about that is to get back to a known kernel without
    editing anything.
    """
    return os.environ.get("HK_AUTOTUNE", "1") not in ("0", "off", "no")


def _record_path(name: str, user: bool = True) -> Path:
    return (records_dir() if user else shipped_dir()) / f"{name}.json"


#: Records read once per process, and the generation they were read at.
#: `schedule_for` is on the launch path -- `hk.attention` resolves a kernel per
#: call -- and a launch path that stats two JSON files is a launch path with a
#: syscall in it. The generation counter is what lets `tune()` write a record
#: and have the next call see it without anyone reloading anything.
_RECORDS: Dict[str, Dict[str, Dict[str, Any]]] = {}
_GEN = 0


def generation() -> int:
    """Bumped whenever a record changes. Caches downstream key on it."""
    return _GEN


def load_records(name: str, *, reload: bool = False) -> Dict[str, Dict[str, Any]]:
    """Shipped records overlaid with user records, keyed by problem key."""
    if not reload and name in _RECORDS:
        return _RECORDS[name]
    out: Dict[str, Dict[str, Any]] = {}
    for user in (False, True):
        try:
            out.update(json.loads(_record_path(name, user).read_text()))
        except (FileNotFoundError, NotADirectoryError, ValueError, OSError):
            # A missing or corrupt record is not worth failing a launch over:
            # the default schedule is a correct kernel, merely perhaps not the
            # fastest one.
            continue
    _RECORDS[name] = out
    return out


def save_record(name: str, key: str, entry: Dict[str, Any]) -> Path:
    global _GEN

    p = _record_path(name)
    cur: Dict[str, Any] = {}
    try:
        cur = json.loads(p.read_text())
    except (FileNotFoundError, ValueError, OSError):
        pass
    cur[key] = entry
    p.write_text(json.dumps(cur, indent=2, sort_keys=True) + "\n")
    _RECORDS.pop(name, None)
    _GEN += 1
    return p


def forget_records() -> None:
    """Drop the in-process copy. For tests, and for a process that knows
    someone else has just tuned underneath it."""
    global _GEN

    _RECORDS.clear()
    _GEN += 1


# ------------------------------------------------------------------ the tuner


#: `None` is a real answer for `Tuner.arch` (a kernel object need not have
#: one), so "not computed yet" needs a value of its own.
_UNSET = object()

#: Every Tuner built at import time, by name. The CLI's whole world.
REGISTRY: Dict[str, "Tuner"] = {}


class Tuner:
    """A kernel factory, a space to search it over, and the record it writes.

    The production path through this object never searches. `kernel(key)` is a
    record lookup and a memoized factory call; with no record for that key it
    returns the default schedule's kernel, which is the one that was shipping
    before the tuner existed. A missing, stale or corrupt record costs
    performance and never correctness.

    `bench` is optional and is only for the CLI: `bench(key) -> fn(kernel) ->
    milliseconds`. Keeping it out of `tune()`'s signature would have been
    tidier, but then `--tune gemm_bf16` would have nowhere to get a benchmark
    from, and a tuner nobody can run from the command line is a tuner that gets
    run from a scratch script instead.
    """

    def __init__(self, name: str, make: Callable[..., Any],
                 schedules: Sequence[Dict[str, Any]],
                 default: Dict[str, Any], *,
                 bench: Optional[Callable[[str], Callable[[Any], float]]] = None,
                 keys: Sequence[str] = (), register: bool = True):
        if not schedules:
            raise ValueError(f"tuner {name}: empty space")
        unknown = [k for s in schedules for k in s if k not in default]
        if unknown:
            raise ValueError(
                f"tuner {name}: schedule axes {sorted(set(unknown))} are not in "
                f"the default schedule {sorted(default)}. The default is the "
                f"full set of knobs: a record that names an axis the default "
                f"does not have cannot be applied to it."
            )
        self.name = name
        self.make = make
        self.schedules = [dict(s) for s in schedules]
        self.default = dict(default)
        self.bench = bench
        self.keys = list(keys)
        self._kernels: Dict[Tuple, Any] = {}
        self._sched_cache: Dict[str, Tuple[int, Dict[str, Any]]] = {}
        self._arch: Any = _UNSET
        if register:
            REGISTRY[name] = self

    # -- production -------------------------------------------------------

    def kernel(self, key: str = ""):
        """The kernel to launch for this problem key.

        A dict probe and a memoized factory call -- no file is read and no
        schedule is rebuilt. That matters because this is in front of every
        `hk.attention` launch, and a 60 us kernel does not have a syscall to
        spare.
        """
        return self._kernel_for(self.schedule_for(key))

    def schedule_for(self, key: str = "") -> Dict[str, Any]:
        """The schedule for this key. The returned dict is shared and cached --
        read it, do not mutate it."""
        hit = self._sched_cache.get(key)
        if hit is not None and hit[0] == generation():
            return hit[1]
        sched = self._schedule_for(key)
        self._sched_cache[key] = (generation(), sched)
        return sched

    @property
    def arch(self) -> Optional[str]:
        """The arch the default kernel targets, if it has one to give.

        Records are tagged with it and a record from another arch is ignored.
        A shipped record travels in a wheel to whatever machine installs the
        wheel, and a schedule measured on gfx1100 says nothing about gfx1201 --
        different register file, different LDS, different answer. Nothing else
        in the lookup path would have caught that, because a key is a problem
        shape and a shape means the same thing on both.
        """
        if self._arch is _UNSET:
            try:
                self._arch = getattr(self._kernel_for(self.default), "arch", None)
            except BaseException:  # noqa: BLE001 -- a tuner must not fail to look up
                self._arch = None
        return self._arch

    def _schedule_for(self, key: str) -> Dict[str, Any]:
        if not enabled():
            return dict(self.default)
        rec = load_records(self.name).get(key)
        if not rec or not rec.get("schedule"):
            return dict(self.default)
        if rec.get("arch") and self.arch and rec["arch"] != self.arch:
            return dict(self.default)
        gain = rec.get("gain")
        if gain is not None and gain < min_gain():
            # Measured, and measured to be a tie. The default is the kernel
            # that was shipping before the tuner existed and the one every
            # other shape here runs; preferring it keeps a serving process on
            # one kernel instead of four chosen by noise.
            return dict(self.default)
        # A record names a schedule by its values and the factory's parameter
        # list is allowed to grow. Unknown keys are dropped rather than passed
        # on: an old record should fall back toward the default, not raise
        # TypeError inside a serving process.
        sched = {k: v for k, v in rec["schedule"].items() if k in self.default}
        return {**self.default, **sched}

    def _kernel_for(self, schedule: Dict[str, Any]):
        k = tuple(sorted(schedule.items()))
        if k not in self._kernels:
            self._kernels[k] = self.make(**schedule)
        return self._kernels[k]

    # -- search -----------------------------------------------------------

    def candidates(self) -> List[Tuple[str, Dict[str, Any]]]:
        return [(label_of(s), dict(s)) for s in self.schedules]

    def build_all(self, *, workers: Optional[int] = None,
                  verbose: bool = False) -> Tuple[Dict[str, Any], List[Row]]:
        """Stage 1: compile the space. No GPU is touched here."""
        from ..runtime.warm import warm_cache

        rows: List[Row] = []
        jobs = []
        kernels: Dict[str, Any] = {}
        scheds: Dict[str, Dict[str, Any]] = {}
        for label, sched in self.candidates():
            scheds[label] = sched
            full = {**self.default, **sched}
            try:
                kernels[label] = self.make(**full)
            except (ValueError, TypeError) as e:
                # The factory's own arithmetic refused it -- warp grid, LDS
                # budget, a split that does not divide. Free, and a real answer.
                rows.append(Row(label, sched, "invalid", reason=str(e)))
                continue
            jobs.append((label, kernels[label], {}))

        res = warm_cache(jobs, workers=workers, verbose=verbose)

        built: Dict[str, Any] = {}
        for label, _, _ in jobs:
            sched = scheds[label]
            if label in res.errors:
                e = res.errors[label]
                status = ("rejected" if isinstance(e, (ResourceError, VerifyError))
                          else "error")
                rows.append(Row(label, sched, status,
                                reason=f"{type(e).__name__}: "
                                       f"{str(e).strip().splitlines()[0]}"))
                continue
            b = res.built[label]
            k0 = b.kernels[0] if b.kernels else None
            rows.append(Row(label, sched, "ok",
                            vgpr=k0.vgpr if k0 else None,
                            occupancy=(k0.occ_by_reg or k0.occupancy) if k0 else None,
                            scratch=k0.scratch if k0 else None))
            built[label] = kernels[label]
        return built, rows

    def tune(self, bench: Optional[Callable[[Any], float]] = None, *,
             key: str = "", rounds: int = 3, workers: Optional[int] = None,
             save: bool = True, verbose: bool = False) -> TuneResult:
        """Stage 1 then stage 2, and write the record.

        `bench(kernel) -> milliseconds` does its own warmup and its own
        synchronisation; this function owns only the *order*, which is the part
        that has to be interleaved to mean anything on this chip.
        """
        if bench is None:
            if self.bench is None:
                raise ValueError(f"tuner {self.name}: no benchmark to run")
            bench = self.bench(key)

        t0 = time.perf_counter()
        built, rows = self.build_all(workers=workers, verbose=verbose)
        by_label = {r.label: r for r in rows}

        order = [l for l, _ in self.candidates() if l in built]
        for r in range(rounds):
            # Reverse on odd rounds. A table takes seconds to minutes and the
            # clock drifts across it; a fixed order hands the whole drift to
            # whichever candidate ran last, and reversing hands every candidate
            # both ends of it.
            for label in (order if r % 2 == 0 else list(reversed(order))):
                try:
                    ms = bench(built[label])
                except BaseException as e:  # noqa: BLE001 -- one bad candidate
                    by_label[label].status = "error"
                    by_label[label].reason = f"{type(e).__name__}: {e}"
                    order = [l for l in order if l != label]
                    break
                by_label[label].times.append(ms)
                if verbose:
                    print(f"  r{r} {label:<36}{ms:9.3f} ms")

        for label in order:
            row = by_label[label]
            if row.times:
                # The minimum, not the mean. Every source of noise on a shared
                # GPU adds time; none of it subtracts.
                row.ms = min(row.times)

        timed = [by_label[l] for l in order if by_label[l].ms is not None]
        best = min(timed, key=lambda r: r.ms) if timed else None

        # The default's own time, measured in the same table. Without it a
        # record says "this schedule took 10.637 ms" and cannot answer the only
        # question that matters when deciding whether to ship it: 10.637
        # against what? A winner that beats the default by 0.1% is a winner by
        # measurement noise, and `--ship` refuses it on this field.
        axes = self.schedules[0].keys()
        default_row = by_label.get(
            label_of({k: self.default[k] for k in axes})
        )
        default_ms = default_row.ms if default_row else None

        res = TuneResult(rows=rows, best=best,
                         seconds=time.perf_counter() - t0, key=key)
        if save and best is not None:
            entry = {
                "arch": self.arch,
                "schedule": best.schedule,
                "ms": round(best.ms, 6),
                "toolchain": toolchain_fingerprint(),
                "candidates": len(rows),
                "measured": len(timed),
            }
            if default_ms is not None:
                entry["default_ms"] = round(default_ms, 6)
                entry["gain"] = round((default_ms - best.ms) / default_ms, 5)
            save_record(self.name, key, entry)
        return res

    # -- AOT --------------------------------------------------------------

    def aot_jobs(self) -> List[Tuple[str, Any, Dict[str, Any]]]:
        """Every schedule this tuner can hand to production, as warm jobs.

        The recorded ones *and* the default: the default is what a key with no
        record gets, so leaving it out would move the JIT cost onto exactly the
        shapes nobody measured.
        """
        wanted = [dict(self.default)]
        for rec in load_records(self.name).values():
            s = {k: v for k, v in (rec.get("schedule") or {}).items()
                 if k in self.default}
            if s:
                wanted.append({**self.default, **s})
        seen, jobs = set(), []
        for s in wanted:
            k = tuple(sorted(s.items()))
            if k in seen:
                continue
            seen.add(k)
            try:
                jobs.append((f"{self.name}:{label_of(s)}", self._kernel_for(s), {}))
            except (ValueError, TypeError):
                continue
        return jobs

    def __repr__(self) -> str:
        return f"<hk.Tuner {self.name} {len(self.schedules)} schedules>"


# ------------------------------------------------------------------ the CLI


def _load_registry() -> Dict[str, "Tuner"]:
    """Import the modules that build tuners. No plugin mechanism on purpose:
    the tuners live next to the kernels they tune."""
    from ..ops import attn, gemm  # noqa: F401 -- importing them registers them
    return REGISTRY


def ship(*, min_gain: Optional[float] = None, dry_run: bool = False) -> int:
    """Promote records measured on this machine into the package (`hk/tuned`).

    The filter is the whole point. A tuning run always produces a winner --
    `min` over a column of noisy numbers is never a tie -- so shipping every
    winner means shipping measurement noise as a finding, and a user on a
    quieter machine then gets a tiling chosen by our fan curve. A record is
    carried only if it beat the *default* by `min_gain`, which is the reason
    `tune()` records `default_ms` beside `ms`.

    Two kinds of record are refused outright rather than shipped with a
    caveat, because a wheel travels further than a cache does:

    * no `default_ms` -- there is no answer to "faster than what?", so there
      is nothing to apply the threshold to;
    * no `arch` -- `_schedule_for` only rejects a foreign record when the
      record says which arch it came from, so an untagged record measured on
      gfx1100 would be applied on gfx1201 by a wheel that reached one.

    Both stay in the user's own records and keep working on this machine.
    Shipping them is what is refused; re-running `--tune` fixes both.
    """
    bar = MIN_GAIN if min_gain is None else min_gain
    reg = _load_registry()
    out = shipped_dir()
    kept_total = dropped_total = 0
    for name in sorted(reg):
        recs = load_records(name, reload=True)
        kept: Dict[str, Dict[str, Any]] = {}
        for key, r in sorted(recs.items()):
            shown = key or "(default)"
            gain = r.get("gain")
            if gain is None and r.get("default_ms") and r.get("ms"):
                gain = (r["default_ms"] - r["ms"]) / r["default_ms"]
            if not r.get("arch"):
                print(f"  {name} {shown}: no arch tag, not shipped "
                      f"-- re-run --tune {name}")
                dropped_total += 1
            elif gain is None:
                print(f"  {name} {shown}: no default_ms, not shipped "
                      f"-- re-run --tune {name}")
                dropped_total += 1
            elif gain >= bar:
                print(f"  {name} {shown}: "
                      f"{label_of(r.get('schedule') or {})} +{gain * 100:.1f}%")
                kept[key] = r
            else:
                print(f"  {name} {shown}: +{gain * 100:.2f}% is under "
                      f"{bar * 100:.1f}%, left on the default")
                dropped_total += 1
        kept_total += len(kept)
        if dry_run:
            continue
        path = out / f"{name}.json"
        if kept:
            out.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(kept, indent=2, sort_keys=True) + "\n")
            tmp.replace(path)
        elif path.exists():
            # A tuner whose every record failed the filter must not keep an
            # older shipped file: that file would be the only thing left
            # claiming a win nothing here can reproduce.
            path.unlink()
    if not dry_run:
        forget_records()
    print(f"{kept_total} record(s) shipped, {dropped_total} refused"
          f"{' (dry run, nothing written)' if dry_run else ''} -> {out}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--list", action="store_true", help="tuners and their records")
    ap.add_argument("--tune", metavar="NAME", help="run the search (needs a GPU)")
    ap.add_argument("--key", default=None,
                    help="problem key for --tune (default: the tuner's own keys)")
    ap.add_argument("--aot", action="store_true",
                    help="prebuild every recorded schedule into the cache")
    ap.add_argument("--ship", action="store_true",
                    help="copy worthwhile records into the package (hk/tuned)")
    ap.add_argument("--min-gain", type=float, default=None,
                    help=f"smallest win --ship will carry, as a fraction "
                         f"(default {MIN_GAIN}); the lookup path applies the "
                         f"same bar, overridable with HK_AUTOTUNE_MIN_GAIN")
    ap.add_argument("--dry-run", action="store_true",
                    help="with --ship, say what would be written")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("-j", "--jobs", type=int, default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    reg = _load_registry()

    if args.aot:
        from ..runtime.warm import warm_cache

        jobs = [j for _, t in sorted(reg.items()) for j in t.aot_jobs()]
        print(f"prebuilding {len(jobs)} schedules")
        res = warm_cache(jobs, workers=args.jobs, verbose=args.verbose)
        print(f"{len(res.built)} built, {len(res.errors)} failed, {res.seconds:.1f}s")
        for label, e in sorted(res.errors.items()):
            print(f"  FAIL {label}: {type(e).__name__}: {e}")
        return 0 if res.ok else 1

    if args.ship:
        return ship(min_gain=args.min_gain, dry_run=args.dry_run)

    if args.tune:
        if args.tune not in reg:
            print(f"no tuner named {args.tune!r}; have {sorted(reg)}")
            return 2
        t = reg[args.tune]
        if t.bench is None:
            print(f"tuner {t.name} has no benchmark; call tune() from Python")
            return 2
        keys = [args.key] if args.key is not None else (t.keys or [""])
        rc = 0
        for key in keys:
            print(f"\n=== {t.name} {key or '(default key)'}")
            res = t.tune(t.bench(key), key=key, rounds=args.rounds,
                         workers=args.jobs, verbose=args.verbose)
            print(res.report())
            rc |= 0 if res.best is not None else 1
        return rc

    for name in sorted(reg):
        t = reg[name]
        recs = load_records(name)
        print(f"{name:<16}{len(t.schedules):>3} schedules, {len(recs)} record(s)")
        for key in sorted(recs):
            r = recs[key]
            ms = r.get("ms")
            ms = f"{ms:.3f} ms" if isinstance(ms, (int, float)) else "?"
            # Whether the record is actually in use, not just on disk. A
            # listing that shows a recorded schedule the lookup path is
            # ignoring is a listing that sends someone to the wrong kernel.
            gain, live = r.get("gain"), t.schedule_for(key)
            note = "" if gain is None else f"  {gain * 100:+.2f}%"
            if live != {**t.default, **{k: v for k, v in
                                        (r.get("schedule") or {}).items()
                                        if k in t.default}}:
                note += "  (not applied)"
            print(f"    {key or '(default key)':<24}"
                  f"{label_of(r.get('schedule') or {})}  {ms}{note}")
    return 0


