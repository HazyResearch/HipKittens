"""A vLLM attention backend that runs on the generated kernels.

This is the part an SDPA drop-in cannot reach. vLLM V1 keeps KV in a paged
cache and dispatches through `AttentionImpl.forward(..., kv_cache,
attn_metadata)`; there is no dense tensor on that path, so the only way into a
decoder is to *be* a backend. Registered as `HK_ATTN` through
`vllm.v1.attention.backends.registry.register_backend`, which is vLLM's own
hook for third-party backends.

    HK_PATCH_VLLM=1 vllm serve <model> --attention-backend HK_ATTN \\
        --no-enable-prefix-caching --enable-chunked-prefill=False

**The two flags are a real restriction, not boilerplate.** This backend
declares its own KV cache layout -- `(2, num_blocks, num_kv_heads, block_size,
head_size)` -- because `gl` derives strides from dims and vLLM's default
`(2, num_blocks, block_size, num_kv_heads, head_size)` cannot yield a
contiguous `(block_size, head_dim)` tile for one head. Having its own layout
means it cannot hand a prefill off to vLLM's Triton kernel, so prefill is
served by the *dense* kernel (`hk.ops.attn`) over the new tokens -- which is
only the whole answer when a prefill request's KV is exactly the tokens in
front of it. Prefix caching and chunked prefill both break that, so both are
refused rather than silently computing attention over part of the context.

What is missing, stated plainly: a paged prefill kernel. With one, both flags
go away. See `python/README.md`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Optional

#: What this backend calls itself in diagnostics.
NAME = "HK_ATTN"

#: What `get_name()` answers, which is not the same thing.
#:
#: `vllm/model_executor/layers/attention/attention.py` does
#: `AttentionBackendEnum[self.attn_backend.get_name()]` -- it round-trips the
#: name back through the enum -- so a backend registered as an override has to
#: answer with the *member it overrides*, not with what it is. Returning
#: "HK_ATTN" here fails at model construction with "Unknown attention
#: backend", which is a clear message for a confusing rule.
_SLOT_NAME = "TRITON_ATTN"

#: The enum member to register under.
#:
#: vLLM's `AttentionBackendEnum` is closed -- `register_backend` records an
#: override for an existing member rather than adding one -- and it does
#: reserve `CUSTOM` for third parties. But `RocmPlatform.get_attn_backend_cls`
#: has an allowlist that `CUSTOM` is not on, so on ROCm selecting it fails
#: with "Attention backend CUSTOM is not supported on ROCm" before any of this
#: code runs. What does work is overriding a member the platform already
#: accepts: `get_path()` consults the override table, so a backend registered
#: under `TRITON_ATTN` is what `--attention-backend TRITON_ATTN` resolves to.
#:
#: That is a loaded gun -- `TRITON_ATTN` is also what ROCm picks by default --
#: so registration is its own opt-in (`HK_VLLM_BACKEND=1`) rather than
#: something `HK_PATCH_VLLM=1` does on the way past. Replacing the decoder
#: attention of a serving engine should take saying so.
SLOT = "TRITON_ATTN"

#: The metadata builder class, made once: vLLM asks for it per KV-cache group.
_BUILDER: list = []


def _build():
    """Construct the backend classes against the installed vLLM.

    Done inside a function, and imported lazily, because this module is
    imported by `hk.integration` on a machine that may have no vLLM at all --
    and because vLLM moves these symbols between releases often enough that an
    import error at the top of the file would take the whole package with it.
    """
    import torch
    from vllm.v1.attention.backend import (
        AttentionBackend,
        AttentionImpl,
        AttentionType,
        MultipleOf,
    )

    from ..ops import attn as hk_attn
    from ..ops import paged as hk_paged

    class HKAttentionBackend(AttentionBackend):
        accept_output_buffer: bool = True
        supported_dtypes: ClassVar[list] = [torch.bfloat16]

        @staticmethod
        def get_name() -> str:
            # The overridden member's name; see _SLOT_NAME.
            return _SLOT_NAME

        @staticmethod
        def get_impl_cls():
            return HKAttentionImpl

        @staticmethod
        def get_builder_cls():
            # The metadata this needs -- block_table, seq_lens, slot_mapping,
            # query_start_loc -- is exactly what the ROCm backend's builder
            # already assembles, and reimplementing it would be copying a file
            # to change nothing in it. Everything but its CUDA-graph claim.
            if _BUILDER:
                return _BUILDER[0]
            from vllm.v1.attention.backends.rocm_attn import (
                RocmAttentionMetadataBuilder,
            )

            class HKAttentionMetadataBuilder(RocmAttentionMetadataBuilder):
                # The ROCm builder claims ALWAYS, which puts `forward` inside
                # a full graph. There, the split count planned on the host
                # from `max_seq_len` is frozen at the capture batch's value,
                # the once-per-layer check syncs mid-capture, and a scratch
                # buffer regrown after capture leaves the graph writing to
                # freed memory. NEVER makes vLLM capture the model piecewise,
                # with attention run eagerly between the pieces.
                _cudagraph_support = type(
                    RocmAttentionMetadataBuilder._cudagraph_support).NEVER

            _BUILDER.append(HKAttentionMetadataBuilder)
            return HKAttentionMetadataBuilder

        @staticmethod
        def get_supported_kernel_block_sizes():
            # The page size *is* the kernel's KV block, so there is exactly one.
            return [hk_paged.KV_BLOCK]

        @classmethod
        def get_supported_head_sizes(cls) -> list[int]:
            return list(hk_attn.HEAD_DIMS)

        @staticmethod
        def get_kv_cache_shape(num_blocks: int, block_size: int,
                               num_kv_heads: int, head_size: int,
                               cache_dtype_str: str = "auto") -> tuple:
            if block_size != hk_paged.KV_BLOCK:
                raise ValueError(
                    f"{NAME} needs block_size {hk_paged.KV_BLOCK}, got "
                    f"{block_size}: the page size is the kernel's KV block, "
                    f"not a tiling choice."
                )
            # Head-major, which is the whole reason this backend exists. See
            # the module docstring.
            return (2, num_blocks, num_kv_heads, block_size, head_size)

        @classmethod
        def validate_head_size(cls, head_size: int) -> None:
            # Defined here rather than inherited: the base class grew this in
            # a later release than the one this was first run against, and a
            # backend that silently accepts a head_dim it has no kernel for
            # fails at launch time with a shape error instead of here with a
            # sentence.
            if head_size not in cls.get_supported_head_sizes():
                raise ValueError(
                    f"{NAME} has kernels for head_size "
                    f"{cls.get_supported_head_sizes()}, not {head_size}. "
                    f"Instantiate one in hk.ops.attn and hk.ops.paged, or run "
                    f"this model on TRITON_ATTN."
                )

        @staticmethod
        def use_cascade_attention(*args, **kwargs) -> bool:
            return False

    class HKAttentionImpl(AttentionImpl):
        def __init__(self, num_heads: int, head_size: int, scale: float,
                     num_kv_heads: int, alibi_slopes, sliding_window,
                     kv_cache_dtype: str, logits_soft_cap=None,
                     attn_type=None, kv_sharing_target_layer_name=None,
                     sinks=None, **kw) -> None:
            unsupported = {
                "alibi_slopes": alibi_slopes,
                "sliding_window": sliding_window,
                "sinks": sinks,
            }
            bad = [k for k, v in unsupported.items() if v]
            if bad:
                raise NotImplementedError(
                    f"{NAME} does not implement {', '.join(bad)}. The kernel "
                    f"is plain causal attention; anything that changes the "
                    f"mask needs the mask written, not a flag threaded."
                )
            if logits_soft_cap:
                raise NotImplementedError(f"{NAME} has no logits_soft_cap")
            if not kv_cache_dtype.startswith("auto"):
                raise NotImplementedError(
                    f"{NAME} keeps the cache in the model dtype, not "
                    f"{kv_cache_dtype}"
                )
            self.num_heads = num_heads
            self.head_size = head_size
            self.scale = float(scale)
            self.num_kv_heads = num_kv_heads
            self.group = num_heads // num_kv_heads
            self._buf_cache = {}
            self._checked = False
            HKAttentionBackend.validate_head_size(head_size)

        # -- the cache -----------------------------------------------------

        def _update_cache(self, key, value, k_cache, v_cache, slot_mapping):
            """Scatter the new tokens into their pages.

            A strided scatter, because the layout is head-major: slot s lives
            at `[s // block, :, s % block, :]`. That is the trade this backend
            makes on purpose -- a decode step reads seq_len pages per layer and
            writes one, so the read is what the layout is chosen for.
            """
            bs = k_cache.shape[2]
            blk = slot_mapping // bs
            pos = slot_mapping % bs
            k_cache[blk, :, pos] = key
            v_cache[blk, :, pos] = value

        # -- the launch ----------------------------------------------------

        def forward(self, layer, query, key, value, kv_cache, attn_metadata,
                    output=None, output_scale=None, output_block_scale=None):
            if attn_metadata is None:                 # profiling run
                return output.fill_(0)
            if output_scale is not None or output_block_scale is not None:
                raise NotImplementedError(f"{NAME} has no fused output quant")

            n = attn_metadata.num_actual_tokens
            k_cache, v_cache = kv_cache[0], kv_cache[1]
            self._update_cache(key[:n], value[:n], k_cache, v_cache,
                               attn_metadata.slot_mapping[:n])

            # The common case, and the one that must not touch the host.
            # `max_query_len` is already a Python int in the metadata, so a
            # pure-decode step can be recognised without reading a device
            # tensor -- and with one token per request, `query` is already in
            # request order, so there is no gather either. The first version
            # of this called .tolist() on two tensors unconditionally, which
            # is a device sync per layer per step: 28 of them on this model,
            # in front of a kernel that takes microseconds.
            if attn_metadata.max_query_len == 1:
                self._decode_all(query, output, k_cache, v_cache,
                                 attn_metadata, n)
                return output

            starts = attn_metadata.query_start_loc
            seq_lens = attn_metadata.seq_lens
            # A step with prefill in it is heavy enough that one host round
            # trip is noise, and what it buys is the ability to refuse a batch
            # the kernel cannot serve instead of answering it wrongly.
            q_lens = (starts[1:] - starts[:-1]).tolist()
            lens = seq_lens.tolist()

            decode = [i for i, q in enumerate(q_lens) if q == 1]
            prefill = [i for i, q in enumerate(q_lens) if q != 1]

            for i in prefill:
                if q_lens[i] != lens[i]:
                    raise NotImplementedError(
                        f"{NAME} has no paged prefill: request {i} has "
                        f"{q_lens[i]} new tokens against a context of "
                        f"{lens[i]}, so part of its KV is in the cache. Run "
                        f"with --no-enable-prefix-caching and chunked prefill "
                        f"off, or implement the paged prefill kernel."
                    )
                lo = int(starts[i])
                self._prefill(query, key, value, output, lo, q_lens[i])

            if decode:
                self._decode(query, output, k_cache, v_cache, attn_metadata,
                             starts, decode)
            return output

        def _prefill(self, query, key, value, output, lo, m):
            """Dense causal attention over this request's own tokens.

            Correct exactly when the request's KV is the tokens in front of
            it, which `forward` has already checked.

            Short prompts go to torch. The dense kernel wants at least one
            192-row query tile and one 32-row KV block, and a prompt below
            that cannot be padded into shape: zero-padded keys are not masked
            out, they are keys whose score is zero, and `exp(0)` is a real
            weight in the softmax sum. Prefilling eight tokens is a rounding
            error of the step's work anyway -- what matters is that the
            fallback is the *reason* the shape is unsupported, checked, rather
            than a length threshold guessed here and left to drift.
            """
            import torch.nn.functional as F

            h, d = self.num_heads, self.head_size
            q = query[lo:lo + m].transpose(0, 1).unsqueeze(0).contiguous()
            k = key[lo:lo + m].transpose(0, 1).unsqueeze(0).contiguous()
            v = value[lo:lo + m].transpose(0, 1).unsqueeze(0).contiguous()
            import os

            if os.environ.get("HK_NO_PREFILL") or hk_attn._why(q, k, v, True):
                out = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=self.scale,
                    enable_gqa=self.group > 1)
            else:
                out = hk_attn.attention(q, k, v, causal=True, scale=self.scale)
            output[lo:lo + m] = out[0].transpose(0, 1).reshape(m, h, d)

        def _buffers(self, n, splits, device, dtype):
            """Scratch for one decode step, allocated once per shape.

            Allocating these per call is 36 allocations and 36 frees per step
            on this model. That is the same class of mistake as the host sync
            that made the first version slower than Triton -- a kernel that
            takes microseconds does not get to be preceded by millisecond
            bookkeeping -- so they are cached on the impl and keyed by the
            shape that decides them.
            """
            import torch

            g, hkv, d = self.group, self.num_kv_heads, self.head_size
            tile = hk_paged.Q_TILE
            # Keyed on `splits`, not on `n`, and grown rather than added to.
            # A decode step's request count moves continuously as a batch
            # ramps -- 1, 3, 7, 13, 16 -- so keying on it means a fresh
            # allocation per distinct count, per split count, per layer, and
            # 36 layers of that against a KV cache already sized to 90% of the
            # card is how a serving process runs itself out of memory in the
            # middle of a run. One buffer per (splits, device, dtype), sized
            # to the largest `n` yet seen and sliced down.
            key = (splits, device, dtype)
            buf = self._buf_cache.get(key)
            if buf is None or buf[0].shape[0] < n:
                cap = max(n, buf[0].shape[0] if buf else 0)
                q = torch.zeros(cap, hkv, tile, d, device=device, dtype=dtype)
                out = torch.empty_like(q)
                if splits > 1:
                    o_part = torch.empty(cap * splits, hkv, tile, d,
                                         device=device, dtype=torch.float32)
                    ml = torch.empty(cap * splits, hkv, 2, tile, device=device,
                                     dtype=torch.float32)
                else:
                    o_part = ml = None
                buf = self._buf_cache[key] = (q, out, o_part, ml)
            q, out, o_part, ml = buf
            # o_part and ml are indexed as `req * splits + split`, so the
            # slice for `n` requests is the first `n * splits` rows -- not the
            # first `n`, which would be the right answer for a different
            # layout and silently wrong for this one.
            return (q[:n], out[:n],
                    None if o_part is None else o_part[:n * splits],
                    None if ml is None else ml[:n * splits])

        def _decode_all(self, query, output, k_cache, v_cache, attn_metadata,
                        n):
            """Every request has exactly one token. No gather, no sync."""
            g, hkv, d = self.group, self.num_kv_heads, self.head_size
            # How empty the machine is decides the schedule. `max_seq_len` is
            # a host int in the metadata, so this costs no sync.
            pages = max(1, -(-attn_metadata.max_seq_len // hk_paged.KV_BLOCK))
            splits = hk_paged.plan_splits(n, hkv, pages)
            q, out, o_part, ml = self._buffers(n, splits, query.device,
                                               query.dtype)
            q[:, :, :g] = query[:n].view(n, hkv, g, d)
            table = attn_metadata.block_table[:n]
            lens = attn_metadata.seq_lens[:n]
            if not self._checked:
                self._checked = True
                import torch as _t

                # Once per layer object, not per step. The kernel reads these
                # two as int32 and indexes memory with the result, so a dtype
                # or a stride that is not what it assumes is not a wrong
                # answer, it is a page fault several kernels later.
                for name, t, want in (("block_table", table, _t.int32),
                                      ("seq_lens", lens, _t.int32)):
                    if t.dtype != want or not t.is_contiguous():
                        raise TypeError(
                            f"{NAME}: {name} is {t.dtype} "
                            f"{'contiguous' if t.is_contiguous() else 'strided'}"
                            f" {tuple(t.shape)}; the kernel reads it as "
                            f"contiguous {want}."
                        )
                mx = int(lens.max())
                cols = table.shape[-1]
                if -(-mx // hk_paged.KV_BLOCK) > cols:
                    raise ValueError(
                        f"{NAME}: seq_len {mx} needs "
                        f"{-(-mx // hk_paged.KV_BLOCK)} pages but block_table "
                        f"has {cols} columns."
                    )
            if splits > 1:
                hk_paged.split_kernel(d, splits)(
                    q, k_cache, v_cache, table, lens, o_part, ml)
                hk_paged.merge_splits(o_part, ml, splits, out)
            else:
                hk_paged.KERNELS[d](q, k_cache, v_cache, table, lens, out)
            output[:n] = out[:, :, :g].reshape(n, hkv * g, d)

        def _decode(self, query, output, k_cache, v_cache, attn_metadata,
                    starts, rows):
            import torch

            g, hkv, d = self.group, self.num_kv_heads, self.head_size
            tile = hk_paged.Q_TILE
            idx = torch.tensor([int(starts[i]) for i in rows],
                               device=query.device)
            # (reqs, kv_heads, tile, d): the GQA group packed into the query
            # tile, zero-padded. The pad rows are computed and dropped -- each
            # query row's output depends on nothing but that row.
            q = torch.zeros(len(rows), hkv, tile, d, device=query.device,
                            dtype=query.dtype)
            q[:, :, :g] = query[idx].view(len(rows), hkv, g, d)
            o = torch.empty_like(q)
            table = attn_metadata.block_table[rows].contiguous()
            lens = attn_metadata.seq_lens[rows].contiguous().to(torch.int32)
            hk_paged.KERNELS[d](q, k_cache, v_cache, table, lens, o)
            output[idx] = o[:, :, :g].reshape(len(rows), hkv * g, d)

    return HKAttentionBackend, HKAttentionImpl


def register(slot: Optional[str] = None) -> str:
    """Register the backend with vLLM. Idempotent."""
    import os

    from vllm.v1.attention.backends.registry import (
        AttentionBackendEnum,
        register_backend,
    )

    want = slot or os.environ.get("HK_VLLM_BACKEND_SLOT", SLOT)
    globals()["_SLOT_NAME"] = want
    backend, impl = _build()
    globals()["HKAttentionBackend"] = backend
    globals()["HKAttentionImpl"] = impl

    member = getattr(AttentionBackendEnum, want, None)
    if member is None:
        raise RuntimeError(
            f"this vLLM has no AttentionBackendEnum.{want}; the members it "
            f"has are {sorted(m.name for m in AttentionBackendEnum)}. Set "
            f"HK_VLLM_BACKEND_SLOT to one of them."
        )
    register_backend(member, f"{__name__}.HKAttentionBackend")
    return f"{NAME} registered over {want} (--attention-backend {want})"
