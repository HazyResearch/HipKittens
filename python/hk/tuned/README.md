# Shipped tuning records

One JSON file per tuner (`gemm_bf16.json`, `attn_fwd_d128.json`, ...), each a
map from problem key to the schedule that measured fastest, the measurement,
and the toolchain fingerprint it was measured under.

These are the **AOT** half of `hk.autotune`: a wheel carries the schedules that
were measured on the arch it names, so a fresh install picks the right tiling
on its first call instead of the default one, and

    python3 -m hk.autotune --aot

can prebuild every one of them into the content-addressed compile cache before
any request arrives. That is what takes hipcc off the serving path.

A record here is a *default*, not an assertion about your machine. Records
written by `python3 -m hk.autotune --tune NAME` land in `$HK_TUNE_DIR` (by
default beside the compile cache) and override these, because whoever ran the
tuner on this machine measured this machine. `HK_AUTOTUNE=0` ignores both and
pins every tuner to its built-in default schedule.

Records do not travel across toolchains. The `toolchain` field is the same
fingerprint the compile cache keys on -- hipcc's version, the contents of
`include/rdna3`, and `hk` itself -- so a record whose fingerprint no longer
matches was measured on a different compiler and is a guess. Nothing refuses
to use a stale record (a stale schedule is a correct kernel, just possibly not
the fastest one); it is recorded so that a surprising number has somewhere to
be traced back to.

## Why there are so few of them

A record lands here only through `python3 -m hk.autotune --ship`, which carries
a key only if its winner beat the *default* schedule by at least `MIN_GAIN`
(1%, `--min-gain` to change it). The same bar is applied again when a record is
read, so a record that measured a tie is inert whether it travelled in a wheel
or was written on this machine; `HK_AUTOTUNE_MIN_GAIN` overrides it.

The bar exists because `min` over a column of noisy numbers always names a
winner. Four keys out of sixteen clear it. Of the GEMM space's four shapes only
4096x4096x4096 does; of attention's twelve, three, all at head_dim 128.

One of those three deserves reading before it is trusted. `attn_fwd_d128` at
n16384 and n65536 reproduces across repeated sweeps -- same schedule, same gain
to within 0.05 points -- so `qk1 pv2` is a real win. `attn_fwd_d128_causal` at
n4096 does not: the gain reproduces (+1.3% to +2.1%) but the winner is a
different schedule every time. What is actually true there is that the default
`qk2 pv2` is the slow one and all three alternatives beat it by about the same
amount; the record names one of them, arbitrarily. See `python/README.md` for
the three-repeat table.

So this directory being nearly empty is the finding, not a gap. A schedule
space whose winners are inside the noise is a space whose default was already
right, and the useful output of the tuner in that case is the evidence that
nothing needed changing.

## Routing tables

`routes_<kind>_<arch>.json` is a different kind of record: not which
schedule a kernel should use, but which kernel -- or the engine's own -- a
call of a given shape should go to. `tools/hk-bench/route_tune.py` writes
them, and `hk.integration.engine.RouteTable` reads them. Each keeps the raw
timings it decided from, the toolchain fingerprint and the margin (a kernel
takes an M only where it beat the engine by `min_gain`). A table is per arch
and per model: a shape it never measured takes the engine's path.
