# gfx1201 Attention (forward)

`fwd/kernel_d64.cpp`: bf16 flash-attention forward, head dimension 64,
non-causal, grouped-query (`ATTN_H` query heads share `ATTN_H_KV` K/V heads).
Outputs O and the per-row logsumexp.

## Use

```
make                                   # build attn_fwd
make test                              # all seven correctness modes vs a CPU reference
make run                               # time it
make EXTRA_HIPFLAGS="-DATTN_N=4096"    # a different problem size (see below)
```

`./attn_fwd iters verify mode` runs directly. With `iters=0` it only verifies.

The problem size is compile-time: `ATTN_B` (batch, 16), `ATTN_H` (64), `ATTN_H_KV`
(8), `ATTN_N` (sequence length, 1024), each overridable with `-D`. `ATTN_N` must
be a multiple of 64.

The correctness modes feed inputs that isolate one part of the computation each.
For example, with Q = 0 the softmax is exactly uniform, and V[k][d] = k%64 shows
which K/V rows each output read. Mode 0 is random data; `main()` lists them all.

## Schedule

Each wave owns a block of 16 query rows and walks the whole sequence in K/V
blocks of 64. The kernel uses **no LDS and no barriers**: waves share nothing.

- **Q** is loaded once, in the layout the WMMA wants, and stays in registers.
- **K** is loaded straight from global into WMMA operand fragments. Head dim is
  the reduction axis and K stores it contiguously, so no transpose is needed.
- **V** needs a transpose (its reduction axis is the sequence), which RDNA4 can
  do only while loading from global (`global_load_tr_b128`), not from LDS. So V
  is also loaded straight into registers.
- **P** (the softmax probabilities) needs no data movement: the accumulator
  layout of `S = K·Qᵀ` is exactly the operand layout `O += V·P` consumes.

The KV loop is software-pipelined over two score buffers, so while the WMMAs
compute `S` for block *j*, the VALU is exponentiating block *j−1* and the
second GEMM accumulates block *j−2* into O:

```
S[j]   = K[j] · Qᵀ              (WMMA)
P[j-1] = exp2(S[j-1] - max)     (VALU, overlapped with the above)
O     += V[j-2] · P[j-2]        (WMMA)
```

The online softmax rescales O lazily. When a new block raises the running row
maximum by less than 8 (in log2 units) across the whole wave, the old maximum is
kept and the rescale of O is skipped. It is applied only when the maximum really
jumps. O and the logsumexp are written straight from the accumulator registers.

## Limits

The block geometry is fixed: 16 query rows per wave, K/V blocks of 64, 8 waves
per workgroup. Every register tile scales with the blocks and the kernel is
register-bound (`Q_BLOCK_SIZE=32` spills). Causal masking is not implemented.
