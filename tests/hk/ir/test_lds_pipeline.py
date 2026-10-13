"""The async-LDS pass: what it derives, and what it refuses.

Every failure in here is one that the C++ compiler accepts, that
`hip_resources.py` passes with ScratchSize 0, and that produces a wrong answer
only sometimes. That is the whole reason the pass exists, so the error paths
are worth more tests than the happy one -- a pass that silently stopped
checking would look exactly like a pass that had nothing to complain about.

The shape of the derivation: LDS retires in issue order, so retiring tile T
means waiting until only the reads issued *after* T are outstanding. `lgkmcnt`
counts individual ds_read instructions, which for a vectorised pair is two per
16x16 base tile -- see hk.lds.lds_loads.

The pass is the C++ hk-lds-pipeline; it records each derived count on the
wait as `hk.lds_wait_for ... {n = K}`, and hk-translate writes it as the
`kittens::lds_wait_for<K>` immediate.
"""

import re

import pytest

import hk
from hk import VerifyError, bf16, fp16, fp32


def _trace(body, **kw):
    k = hk.kernel(body, arch="gfx1100", warps=1,
                  grid=lambda p: (1, 1, 1), name=body.__name__)
    return k.trace(**kw)


def _src(body):
    return hk.kernel(body, arch="gfx1100", warps=1,
                     grid=lambda p: (1, 1, 1), name=body.__name__).source()


def _pair(rows=32, cols=64, dtype=bf16, layout="row"):
    """(shared tile value, register tile type) that takes the vectorised path."""
    return hk.lds.alloc(hk.ST(dtype, rows, cols)), hk.RT(dtype, rows, cols, layout)


def _waits(traced):
    """The derived count of every lds_wait_for in the kernel body, innermost
    regions included, in program order."""
    out = []
    for ln in traced.dump().splitlines():
        if "hk.lds_wait_for" not in ln:
            continue
        m = re.search(r"\{n = (\d+) : i64\}", ln)
        assert m, f"lds_wait_for without a derived count: {ln}"
        out.append(int(m.group(1)))
    return out


# -- the count ---------------------------------------------------------------


def test_reads_per_tile_is_two_per_base_tile():
    # The number the whole derivation is denominated in. 32x64 bf16 is 2x4
    # base tiles, two ds_read_b128 apiece.
    assert hk.lds.lds_loads(hk.RT(bf16, 32, 64, "row"), hk.ST(bf16, 32, 64)) == 16
    assert hk.lds.lds_loads(hk.RT(bf16, 16, 16, "row"), hk.ST(bf16, 16, 16)) == 2
    # And zero for every pair that misses the vectorised path, which is what
    # load(wait=False) keys its refusal off.
    assert hk.lds.lds_loads(hk.RT(bf16, 32, 64, "col"), hk.ST(bf16, 32, 64)) == 0
    assert hk.lds.lds_loads(hk.RT(fp32, 32, 64, "row"), hk.ST(fp32, 32, 64)) == 0
    assert hk.lds.lds_loads(hk.RT(fp16, 32, 64, "row"), hk.ST(bf16, 32, 64)) == 0


def test_n_counts_only_the_reads_issued_after_the_named_tile():
    def two_in_flight(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)       # 16 reads
        sb, tb = _pair(16, 32)       # 2x2 base tiles -> 4 reads
        a = sa.load(ta, wait=False)
        b = sb.load(tb, wait=False)
        hk.lds.wait_for(a)           # b is still out: lgkmcnt(4)
        hk.lds.wait_for(b)           # nothing left: lgkmcnt(0)

    n = _waits(_trace(two_in_flight))
    assert n == [4, 0]


def test_naming_both_tiles_waits_for_the_later_one():
    def both(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        b = sb.load(tb, wait=False)
        hk.lds.wait_for(a, b)

    assert _waits(_trace(both)) == [0]


def test_the_count_reaches_the_generated_source():
    def emit(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        b = sb.load(tb, wait=False)
        hk.lds.wait_for(a)
        hk.lds.wait_for(b)

    src = _src(emit)
    assert "kittens::load<false>(" in src
    line = [ln.strip() for ln in src.splitlines() if "lds_wait_for" in ln]
    assert len(line) == 2, line
    # a retires with b's four reads still out; b then retires with none.
    assert line[0].startswith("kittens::lds_wait_for<4>("), line
    assert line[1].startswith("kittens::lds_wait_for<0>("), line


# -- the two things it refuses -----------------------------------------------


def test_reading_a_tile_whose_reads_are_in_flight():
    def early(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        a = sa.load(ta, wait=False)
        b = hk.zeros(hk.RT(bf16, 64, 64, "row"))
        hk.mma("mma_ABt", a, b, hk.zeros(hk.RT(fp32, 32, 64, "col")))

    with pytest.raises(VerifyError, match="still in flight"):
        _trace(early)


def test_reading_a_tile_someone_elses_wait_retired():
    """The one that costs days. The data has landed; the dependence has not.

    `lds_wait_for(b)` waits down to lgkmcnt(0) as far as `a` is concerned, so a
    consumer of `a` reads correct memory -- until the scheduler moves it above
    the wait, which nothing forbids, and which it starts doing when register
    pressure shifts somewhere else in the kernel.
    """
    def retired_not_bound(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        b = sb.load(tb, wait=False)
        hk.lds.wait_for(b)           # retires a's reads too, binds only b
        a.copy(into=a)

    with pytest.raises(VerifyError) as e:
        _trace(retired_not_bound)
    msg = str(e.value)
    assert "did not name it" in msg and "register dependence" in msg
    # It has to say which wait to fix, not only that one is wrong.
    assert "Add " in msg and "to that lds_wait_for" in msg


def test_a_synchronous_load_retires_everything_and_binds_nothing():
    """`kittens::load<true>` ends in a bare lgkmcnt(0). Same trap, no wait op.

    This is the version with no `lds_wait_for` anywhere to point at, which is
    why it is worth its own test: the offending op is an innocuous-looking
    second load.
    """
    def sync_in_the_middle(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        sb.load(tb)                  # wait=True: lgkmcnt(0), binds only itself
        a.copy(into=a)

    with pytest.raises(VerifyError, match="register dependence"):
        _trace(sync_in_the_middle)


def test_waiting_for_a_tile_that_has_nothing_outstanding():
    def dead_wait(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        a = sa.load(ta)              # already retired and bound
        hk.lds.wait_for(a)

    with pytest.raises(VerifyError, match="no LDS reads outstanding"):
        _trace(dead_wait)


def test_binding_late_is_legal_and_free():
    """Naming a tile some earlier wait already retired is the fix, not a fault.

    The hazard is never that the data has not landed -- the earlier wait saw
    to that, and lgkmcnt only shrinks in between. It is that the register
    dependence is missing, so the consumer can be scheduled above the point
    where it became safe. A later `lds_wait_for` that names the tile puts that
    edge back: `s_waitcnt` (a no-op by then) followed by the bind, after which
    nothing can move up past it.

    The rotated GEMM does exactly this on most of its chunks. A is reloaded at
    the end of a slice, so it sits behind B chunks 0..n-2 in the queue; the
    next slice's first wait names A and so retires those B chunks early, and
    each is bound at its own chunk's wait. The C++ says so in as many words:
    "A was issued after B chunks 0..n-2 and before B chunk n-1, so for every
    chunk but the last, A is the read that gates."
    """
    def late(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        b = sb.load(tb, wait=False)
        hk.lds.wait_for(b)           # retires a's reads too, binds only b
        hk.lds.wait_for(a)           # catches up: waits for nothing, binds a
        a.copy(into=a)

    # The second wait found an empty queue, so it costs an lgkmcnt(0) that is
    # already satisfied -- the bind is the whole instruction that matters.
    assert _waits(_trace(late)) == [0, 0]


def test_reading_without_ever_binding_is_still_the_error():
    """The relaxation above is only about *waits*. Drop the catch-up wait and
    the consumer is back to reading a tile nothing bound."""
    def unbound(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        b = sb.load(tb, wait=False)
        hk.lds.wait_for(b)
        a.copy(into=a)

    with pytest.raises(VerifyError, match="register dependence"):
        _trace(unbound)


def test_overwriting_a_tile_that_is_still_being_read_into():
    def clobber(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb = hk.lds.alloc(hk.ST(bf16, 32, 64), "b")
        a = sa.load(ta, wait=False)
        sb.load(ta, into=a, wait=False)

    with pytest.raises(VerifyError, match="while its previous LDS reads"):
        _trace(clobber)


def test_the_overwrite_is_diagnosed_as_a_write_not_a_read():
    """`into=` makes the destination an operand of the load, which the
    in-flight *read* check would otherwise reach first.

    It was an error either way -- nothing silent escaped -- but the message
    said "name it in an lds_wait_for before using it" about a tile that is not
    being used, which sends the reader to the wrong fix.
    """
    def clobber(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb = hk.lds.alloc(hk.ST(bf16, 32, 64), "b")
        a = sa.load(ta, wait=False)
        sb.load(ta, into=a, wait=False)

    with pytest.raises(VerifyError) as e:
        _trace(clobber)
    assert "load_shared writes" in str(e.value)
    assert "before using it" not in str(e.value)


def test_an_in_place_consumer_is_still_a_read():
    """The exception above is for load_shared only. `a *= x` reads `a`
    through the same operand it writes and must not get the write
    treatment."""
    def in_place_use(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        a = sa.load(ta, wait=False)
        a *= 2.0
        hk.lds.wait_for(a)

    with pytest.raises(VerifyError, match="still in flight"):
        _trace(in_place_use)


def test_a_kernel_may_not_end_with_reads_in_flight():
    def leaked(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sa.load(ta, wait=False)

    with pytest.raises(VerifyError, match="ends with LDS reads still in flight"):
        _trace(leaked)


def test_an_unvectorizable_pair_cannot_be_issued_asynchronously():
    """Refused at the op, not at the pass: there is nothing to keep in flight.

    A col-layout or 32-bit destination goes through the elementwise fallback,
    whose waits the compiler places itself -- so `wait=False` would be a lie
    rather than a race.
    """
    def col_async(o: hk.Tensor[fp32]):
        s, t = _pair(32, 64, layout="col")
        s.load(t, wait=False)

    with pytest.raises(TypeError, match="does not take the vectorised path"):
        _trace(col_async)


def test_a_wait_must_name_something():
    with pytest.raises(TypeError, match="needs the tiles it retires"):
        hk.lds.wait_for()


# -- loops -------------------------------------------------------------------


def test_the_prefetch_loop_has_one_count_for_every_iteration():
    """The shape a GEMM K-loop actually has: wait for this slice, issue the
    next, do the math on this one. The queue is non-empty across the back
    edge, which is the case a single forward walk gets wrong."""
    def prefetch(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        a = sa.load(ta, wait=False)
        acc = hk.zeros(hk.RT(fp32, 32, 64, "col"))
        b = hk.zeros(hk.RT(bf16, 64, 64, "row"))
        for _ in hk.range(8):
            hk.lds.wait_for(a)
            acc += a @ b.T
            sa.load(ta, into=a, wait=False)
        hk.lds.wait_for(a)

    ir = _trace(prefetch)
    assert "scf.for" in ir.dump()
    assert _waits(ir) == [0, 0]


def test_a_loop_whose_queue_depth_drifts_is_an_error():
    """Two tiles prefetched, one recycled: the first iteration waits with the
    other still outstanding and later ones do not. Either lgkmcnt would be
    wrong for some iteration, so the pass reports instead of picking."""
    def no_steady_state(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        b = sb.load(tb, wait=False)
        for _ in hk.range(8):
            hk.lds.wait_for(a)
            sa.load(ta, into=a, wait=False)
        hk.lds.wait_for(a, b)

    with pytest.raises(VerifyError) as e:
        _trace(no_steady_state)
    assert "no steady state" in str(e.value)
    assert "Peel the prologue" in str(e.value)


def test_a_tile_read_before_the_loops_wait_is_still_caught():
    """The check does not stop at region boundaries."""
    def inside(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        acc = hk.zeros(hk.RT(fp32, 32, 64, "col"))
        b = hk.zeros(hk.RT(bf16, 64, 64, "row"))
        a = sa.load(ta, wait=False)
        for _ in hk.range(8):
            acc += a @ b.T            # no wait anywhere
        hk.lds.wait_for(a)

    with pytest.raises(VerifyError, match="still in flight"):
        _trace(inside)


# -- conditional issue -------------------------------------------------------


def test_a_counted_wait_may_not_count_conditional_ops():
    """`lgkmcnt(n)` counts what *this* wave issued.

    A wave that took the other side of an `if` on a device value has fewer
    reads in flight, so the same `lgkmcnt(n)` lets it run on past reads that
    have not landed -- on some waves, on some schedules. The branch itself is
    fine; counting across it is not.
    """
    def conditional_tail(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        if hk.arch.block_idx.x > 0:
            sb.load(tb, wait=False)
        hk.lds.wait_for(a)           # would be lgkmcnt(4), but only sometimes
        a.copy(into=a)

    with pytest.raises(VerifyError, match="issued inside an hk.if_"):
        _trace(conditional_tail)


def test_the_error_names_both_ways_out():
    def conditional_tail(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        if hk.arch.block_idx.x > 0:
            sb.load(tb, wait=False)
        hk.lds.wait_for(a)
        a.copy(into=a)

    with pytest.raises(VerifyError) as e:
        _trace(conditional_tail)
    msg = str(e.value)
    assert "Retire the conditional ops before the branch" in msg
    assert "barrier(drain=True)" in msg


def test_a_conditional_op_the_wait_does_not_count_is_fine():
    """The rule is about the *tail*. Ops issued before the named tile are
    retired by the wait regardless of how many of them there were, because
    `lgkmcnt` counts down to a depth rather than counting them off."""
    def conditional_head(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        if hk.arch.block_idx.x > 0:
            sb.load(tb, wait=False)
        a = sa.load(ta, wait=False)
        hk.lds.wait_for(a)           # lgkmcnt(0): nothing after a
        a.copy(into=a)

    _trace(conditional_head)


def test_a_counted_wait_inside_the_branch_may_count_that_branch():
    """Every wave that reaches a wait inside an `if` entered it and issued
    what came before the wait in it, so counting those is the same number on
    every wave. This is the batched read of a combine done by one warp."""
    def same_branch(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        if hk.arch.block_idx.x > 0:
            a = sa.load(ta, wait=False)
            b = sb.load(tb, wait=False)
            hk.lds.wait_for(a)           # lgkmcnt(4): b, issued in this branch
            a.copy(into=a)
            hk.lds.wait_for(b)
            b.copy(into=b)

    ir = _trace(same_branch)
    assert "scf.if" in ir.dump()
    assert _waits(ir) == [4, 0]


def test_an_enclosing_branch_that_is_still_open_counts_too():
    def nested(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        if hk.arch.block_idx.x > 0:
            a = sa.load(ta, wait=False)
            b = sb.load(tb, wait=False)
            if hk.arch.block_idx.y > 0:
                hk.lds.wait_for(a)       # b came from the enclosing branch
                a.copy(into=a)
            hk.lds.wait_for(a, b)
            b.copy(into=b)

    _trace(nested)


def test_a_branch_that_has_closed_may_not_be_counted_from_its_parent():
    def closed_inner(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        if hk.arch.block_idx.x > 0:
            a = sa.load(ta, wait=False)
            if hk.arch.block_idx.y > 0:
                sb.load(tb, wait=False)
            hk.lds.wait_for(a)           # would count b, which only some issued
            a.copy(into=a)

    with pytest.raises(VerifyError, match="issued inside an hk.if_"):
        _trace(closed_inner)


def test_last_iterations_branch_is_closed_even_if_it_is_the_same_branch():
    """The same static `if` inside a loop is a new branch every iteration.
    Reads it left in flight last time are from a branch that has closed."""
    def across_iterations(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        b = hk.zeros(tb)
        for i in hk.range(4):
            if i > 0:
                hk.lds.wait_for(a)
                a.copy(into=a)
                sa.load(ta, wait=False, into=a)
                sb.load(tb, wait=False, into=b)

    with pytest.raises(VerifyError, match="issued inside an hk.if_"):
        _trace(across_iterations)


def test_a_draining_barrier_does_not_count_and_so_is_allowed():
    """`barrier(drain=True)` is lgkmcnt(0), which is the same instruction on
    every wave however many ops it issued."""
    def drained(o: hk.Tensor[fp32]):
        sa, ta = _pair(32, 64)
        sb, tb = _pair(16, 32)
        a = sa.load(ta, wait=False)
        if hk.arch.block_idx.x > 0:
            sb.load(tb, wait=False)
        hk.arch.barrier(drain=True)
        hk.lds.wait_for(a)           # a pure re-bind now: nothing outstanding
        a.copy(into=a)

    _trace(drained)
