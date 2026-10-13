"""Derive every `s_waitcnt lgkmcnt` count, and prove every async tile is bound.

This pass is the reason the IR exists.

`load_shared(wait=False)` issues its ds_reads through an inline-asm *output*
operand, so LLVM believes the destination is ready the instant the asm ends.
`s_waitcnt` is a separate volatile asm sharing no operands with it, and
`v_wmma` is a pure builtin with no memory effect. Nothing in the IR stops the
scheduler from hoisting a consuming WMMA above the wait, and it does --
nondeterministically, as soon as register pressure shifts. It is not a spill:
ScratchSize stays 0 while the answer changes from run to run. The fix is to
bind the tile's registers after the wait, which `kittens::lds_wait_for` does;
the problem is that a human writing it by hand forgets, and nothing tells them.

So three things are checked here, and they are errors, not warnings:

  * a tile whose reads are outstanding may not be read;
  * a tile whose reads have been retired by somebody else's wait may not be
    read either, because retiring is not binding; and
  * a barrier may not be reached with LDS ops in flight, because `s_barrier`
    orders execution and not memory.

And one thing is *computed*, so that it cannot be got wrong: the `N` in
`lgkmcnt(N)`. LDS returns in order, so retiring tile T means waiting until only
the reads issued after T are outstanding. The handwritten GEMM derives exactly
this number with a constexpr function and a paragraph explaining why
(`CHUNK_READS` in kernels/rdna3/gemm/bf16fp32/gemm.cpp); here it falls out of
walking the issue queue, and `lds_wait_for` has no `n` parameter to get wrong.

**Loops.** A prefetching loop issues reads for iteration k+1 and waits for
iteration k's at the top of the *next* iteration, so the queue is non-empty
across the back edge and the derived N depends on state from the previous
iteration. The pass therefore walks each loop body twice, the second time
starting from the state the first left, and requires the two to agree. A body
whose derived counts differ between the two passes has no steady state -- the
queue grows or shrinks each iteration -- and that is reported rather than
guessed at, because either answer would be wrong for some iteration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ..nodes import KernelIR, Op, RegTileType, Value
from ..verify import VerifyError


def _at(op: Op) -> str:
    return f"\n    at {op.loc}" if op.loc else ""


@dataclass
class _State:
    """Where the LDS read queue stands at a point in the program.

    `queue` is outstanding reads in issue order -- LDS retires in that order,
    which is the only reason a count is derivable at all. `unbound` is tiles
    whose reads have been retired by a wait that did not name them: no longer
    in flight, still missing the register dependence, still unusable.
    """

    #: Outstanding LDS ops in issue order: (id, count, value, kind, scope).
    #: `kind` is "read" or "write" -- lgkmcnt counts both, and a schedule that
    #: leaves ds_writes in flight across a wait has to carry them in the
    #: immediate or it retires more reads than it meant to. The handwritten
    #: GEMM calls that term TAIL_OPS and computes it by hand; here it is just a
    #: queue entry.
    #:
    #: `scope` is the stack of `hk.if_` regions the op was issued in -- () for
    #: unconditional -- or None once one of those regions has closed. A wave
    #: that took the other side of a closed branch never issued the op, and
    #: `lgkmcnt(n)` is a number, not a set, so a *counted* wait with such an op
    #: in its tail waits down to the wrong depth on that wave. An op whose
    #: regions are all still open around the wait is different: every wave
    #: that reaches the wait entered them and issued it. Only the tail matters
    #: -- see the wait handler.
    queue: List[Tuple[int, int, Value, str, Optional[tuple]]] = field(
        default_factory=list)
    #: Tiles a wait has retired *and* named, so their register dependence is
    #: in place. Naming one again is legal and free; see the wait handler.
    bound: Dict[int, Value] = field(default_factory=dict)
    unbound: Dict[int, Op] = field(default_factory=dict)  # id -> the wait that retired it

    def copy(self) -> "_State":
        return _State(queue=list(self.queue), bound=dict(self.bound),
                      unbound=dict(self.unbound))

    def index_of(self, v: Value) -> Optional[int]:
        for i, (vid, _, _, kind, _c) in enumerate(self.queue):
            if vid == id(v) and kind == "read":
                return i
        return None


def _check_reads(op: Op, st: _State) -> None:
    """Nothing may read a tile that is in flight or retired-but-unbound."""
    for v in op.operands:
        if not isinstance(v.type, RegTileType):
            continue
        if st.index_of(v) is not None:
            raise VerifyError(
                f"{op.opcode} reads {v.name}, whose LDS reads are still in "
                f"flight. Name it in an lds_wait_for before using it. This is "
                f"an error rather than a race you might get away with: the "
                f"reads have not landed, and the consumer will not wait for "
                f"them on its own." + _at(op)
            )
        w = st.unbound.get(id(v))
        if w is not None:
            raise VerifyError(
                f"{op.opcode} reads {v.name}. Its LDS reads were retired by an "
                f"lds_wait_for that did not name it, so the data has landed but "
                f"the register dependence has not been put back -- this "
                f"consumer is free to schedule above the wait, and will, "
                f"intermittently. Add {v.name} to that lds_wait_for."
                + _at(w) + _at(op)
            )


def _live(issued: Optional[tuple], scope: tuple) -> bool:
    """Whether every wave that reaches `scope` issued an op issued in `issued`."""
    return issued is not None and scope[:len(issued)] == issued


def _walk(ops: List[Op], st: _State, second: bool, scope: tuple = ()) -> _State:
    for op in ops:
        if op.body is not None:
            if op.opcode == "if":
                # Once. A branch body runs at most once per encounter, so the
                # steady state a loop has to converge to does not exist here,
                # and walking twice makes the second pass see the first pass's
                # own conditionally-issued reads still in flight -- which gets
                # reported as "writes a tile whose reads are in flight", a
                # confident diagnosis of something the kernel never does. The
                # tail rule below is what actually covers a branch: ops issued
                # inside one carry it in their scope, and once it closes, a
                # *counted* wait that would have to count them is the error.
                region = id(op)
                st = _walk(op.body, st, second, scope + (region,))
                st.queue = [(vid, c, v, kind,
                             None if s is None or region in s else s)
                            for vid, c, v, kind, s in st.queue]
            else:
                # A loop. Run it twice and require agreement; see the module
                # docstring.
                after_first = _walk(op.body, st.copy(), second, scope)
                st = _walk(op.body, after_first, True, scope)
            continue

        if op.opcode == "lds_wait_for":
            positions = []
            for v in op.operands:
                i = st.index_of(v)
                if i is None:
                    if id(v) in st.bound or id(v) in st.unbound:
                        # Already retired by some earlier wait, with or without
                        # a binding. Either way naming it here is correct and
                        # costs nothing: this wait emits an `s_waitcnt` that is
                        # at least as strong as the one that retired the reads
                        # -- the queue only shrinks between them -- and then
                        # binds the fragment, so no consumer can be scheduled
                        # above *this* point. It contributes no position and is
                        # left out of the cut.
                        #
                        # Both cases are routine in the rotated schedule, and
                        # the second is the one that matters. A is reloaded at
                        # the end of a slice, so it sits behind B chunks
                        # 0..n-2 in the queue; the next slice's first wait
                        # names A, which retires those B chunks early without
                        # naming them. Each is then bound at its own chunk's
                        # wait, which is exactly here. Rejecting that would
                        # mean the author has to work out which half of an
                        # operand pair is still moving before they can name
                        # it -- the bookkeeping this pass exists to take over.
                        # A consumer that reads such a tile *without* any wait
                        # naming it is still an error; _check_reads says so.
                        continue
                    raise VerifyError(
                        f"lds_wait_for names {v.name}, which has no LDS reads "
                        f"outstanding and has never had any. Either the load "
                        f"is missing, or this names the wrong tile."
                        + _at(op)
                    )
                positions.append(i)

            # No position at all means every named tile was already retired and
            # bound: a pure re-bind. It waits down to the depth it found, which
            # is to say it waits for nothing.
            cut = max(positions) if positions else -1
            # Everything issued after the last-named tile stays in flight, and
            # its read count is exactly the lgkmcnt to wait down to.
            n = sum(c for _, c, _, _, _s in st.queue[cut + 1:])
            if n and not all(_live(s, scope) for _, _, _, _, s in st.queue[cut + 1:]):
                raise VerifyError(
                    f"this lds_wait_for needs lgkmcnt({n}), but some of the "
                    f"LDS ops it is counting were issued inside an hk.if_. A "
                    f"wave that took the other side of the branch has fewer in "
                    f"flight and this wait lets it run on past reads that have "
                    f"not landed. Retire the conditional ops before the branch "
                    f"closes, or use barrier(drain=True), which is lgkmcnt(0) "
                    f"and does not count." + _at(op)
                )
            named = {id(v) for v in op.operands}
            for vid, _, v, kind, _c in st.queue[: cut + 1]:
                if kind == "write":
                    # A retired write needs no binding: nothing reads the
                    # registers it came from, and the LDS it wrote is read
                    # through a fresh ds_read that this wait has ordered.
                    continue
                if vid not in named:
                    # Retired by this wait but not bound by it. Legal to write,
                    # illegal to read -- _check_reads says so at the use.
                    st.unbound[vid] = op
            st.queue = st.queue[cut + 1:]
            for v in op.operands:
                st.unbound.pop(id(v), None)
                st.bound[id(v)] = v

            if "n" in op.attrs and op.attrs["n"] != n:
                raise VerifyError(
                    f"this lds_wait_for needs lgkmcnt({op.attrs['n']}) on one "
                    f"iteration and lgkmcnt({n}) on the next: the loop has no "
                    f"steady state, so the queue depth depends on how many "
                    f"iterations have run. Peel the prologue out of the loop, "
                    f"or issue the same reads every iteration." + _at(op)
                )
            op.attrs["n"] = n
            continue

        if op.opcode == "barrier":
            # s_barrier synchronises execution, not memory. An LDS op still in
            # flight here is a race with whatever the *other* warps do after
            # the barrier, and on a double-buffered kernel that is always the
            # opposite access: outstanding ds_writes race the next tile's
            # ds_reads, outstanding ds_reads race the next tile's ds_writes.
            # The handwritten GEMM writes `lds_wait<0>()` before its barrier
            # with a comment saying exactly this; getting it wrong produces a
            # kernel that is correct until the scheduler moves something.
            if st.queue:
                if not op.attrs.get("drain"):
                    kinds = sorted({k for _, _, _, k, _c in st.queue})
                    what = " and ".join(f"ds_{k}s" for k in kinds)
                    raise VerifyError(
                        f"barrier with {what} still in flight. s_barrier orders "
                        f"execution, not memory: past it the other warps touch "
                        f"this LDS the other way round, and nothing waits for "
                        f"these. Either retire them first -- an lds_wait_for "
                        f"that names the tiles, which is free if the math "
                        f"needed them anyway -- or write barrier(drain=True) to "
                        f"spend an lgkmcnt(0) here." + _at(op)
                    )
                # drain=True is lgkmcnt(0): it retires everything and binds
                # nothing, exactly like load<true>.
                for vid, _, _, kind, _c in st.queue:
                    if kind == "read":
                        st.unbound[vid] = op
                st.queue = []
            continue

        if op.opcode in ("load_shared", "load_frag"):
            # Before _check_reads, and that order is load-bearing. `out=`
            # compiles to emit_into, which puts the destination at operands[0]
            # -- so a load into an in-flight tile looks to _check_reads like a
            # *read* of one, and gets diagnosed as a missing wait before use
            # when the actual fault is that the previous reads have not landed
            # yet and will overwrite the new ones. Same verdict, wrong
            # instruction to the reader. load_shared is the only op here whose
            # first operand is written without being read; an in-place copy or
            # mul really does read it, and must go through _check_reads.
            dst = op.dst
            if st.index_of(dst) is not None:
                raise VerifyError(
                    f"{op.opcode} writes {dst.name} while its previous LDS "
                    f"reads are still in flight. The earlier reads would land "
                    f"on top of the newer ones -- retire them with an "
                    f"lds_wait_for first." + _at(op)
                )
            if op.attrs.get("wait", True):
                # kittens::load<true> ends in lgkmcnt(0), which retires every
                # outstanding read, not only this tile's -- and binds none of
                # them. Anything else in the queue becomes unbound.
                for vid, _, _, kind, _c in st.queue:
                    if kind == "read":
                        st.unbound[vid] = op
                st.queue = []
                st.unbound.pop(id(dst), None)
            else:
                st.unbound.pop(id(dst), None)
                st.bound.pop(id(dst), None)
                st.queue.append((id(dst), op.attrs["lds_loads"], dst,
                                 "read", scope))
            continue

        if op.opcode == "store_frag":
            # Two raw ds_write_b128 with no wait of their own -- unlike
            # kittens::store, which ends in lgkmcnt(0). Nothing ever names them
            # in a wait; they are here only to raise the immediate of every
            # wait issued while they are in flight.
            st.queue.append((id(op), 2, op.operands[0], "write", scope))
            continue

        if op.opcode == "stage_commit":
            # ds_writes. They share lgkmcnt with the reads, so they belong in
            # the same queue even though nothing ever names them in a wait --
            # their only effect is to raise the immediate of every wait issued
            # while they are in flight.
            dst, buf = op.operands
            if op.attrs.get("wait"):
                for vid, _, _, kind, _c in st.queue:
                    if kind == "read":
                        st.unbound[vid] = op
                st.queue = []
            else:
                st.queue.append((id(op), buf.type.calls, dst, "write", scope))
            continue

        _check_reads(op, st)
    return st


def lds_pipeline(ir: KernelIR) -> None:
    final = _walk(ir.body, _State(), False)
    reads = [v for _, _, v, kind, _c in final.queue if kind == "read"]
    if reads:
        names = ", ".join(v.name for v in reads)
        raise VerifyError(
            f"the kernel ends with LDS reads still in flight for {names}. "
            f"Either they are dead -- drop the load -- or a wait is missing."
        )
    # Trailing ds_writes are not an error. The kernel is about to end, and the
    # hardware retires them; nothing in this workgroup reads them again.
