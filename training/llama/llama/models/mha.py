# Copyright (c) 2023, Tri Dao.
import torch
import torch.nn as nn
from einops import rearrange
    
try:
    from .rotary import RotaryEmbedding
except ImportError:
    RotaryEmbedding = None


from .attentions.base import SelfAttention, CrossAttention
from .attentions.aiter import AITERSelfAttention, AITERCrossAttention
from .attentions.hipkittens import HipSelfAttention, HipCrossAttention


class LinearResidual(nn.Linear):
    """Wrap nn.Linear to return the residual as well. For compatibility with FusedDense."""

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super().forward(input), input


def _update_kv_cache(kv, inference_params, layer_idx):
    """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
    # Pre-allocate memory for key-values for inference.
    num_heads, head_dim = kv.shape[-2:]
    if layer_idx not in inference_params.key_value_memory_dict:
        kv_cache = torch.empty(
            inference_params.max_batch_size,
            inference_params.max_seqlen,
            2,
            num_heads,
            head_dim,
            dtype=kv.dtype,
            device=kv.device,
        )
        inference_params.key_value_memory_dict[layer_idx] = kv_cache
    else:
        kv_cache = inference_params.key_value_memory_dict[layer_idx]
    # Adjust key and value for inference
    batch_start = inference_params.batch_size_offset
    batch_end = batch_start + kv.shape[0]
    sequence_start = inference_params.seqlen_offset
    sequence_end = sequence_start + kv.shape[1]
    assert batch_end <= kv_cache.shape[0]
    assert sequence_end <= kv_cache.shape[1]
    assert kv_cache is not None
    kv_cache[batch_start:batch_end, sequence_start:sequence_end, ...] = kv
    return kv_cache[batch_start:batch_end, :sequence_end, ...]


class MHA(nn.Module):
    """Multi-head self-attention and cross-attention"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        qkv_proj_bias=True,
        out_proj_bias=True,
        dropout=0.0,
        softmax_scale=None,
        causal=False,
        layer_idx=None,
        rotary_emb_dim=0,
        rotary_emb_base=10000.0,
        rotary_emb_scale_base=None,
        rotary_emb_interleaved=False,
        fused_bias_fc=False,
        use_aiter_attn=False,
        use_hip_attn=False,
        return_residual=False,
        checkpointing=False,
        device=None,
        dtype=None,
        **kwargs,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.embed_dim = embed_dim
        self.causal = causal
        self.layer_idx = layer_idx
        self.rotary_emb_dim = rotary_emb_dim
        self.return_residual = return_residual
        self.checkpointing = checkpointing

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        kv_dim = 2 * self.head_dim * self.num_heads_kv

        if self.rotary_emb_dim > 0:
            assert RotaryEmbedding is not None, "rotary_emb is not installed"
            self.rotary_emb = RotaryEmbedding(
                self.rotary_emb_dim,
                base=rotary_emb_base,
                scale_base=rotary_emb_scale_base,
                interleaved=rotary_emb_interleaved,
                device=device,
            )

        linear_cls = nn.Linear
        linear_resid_cls = (LinearResidual)
        wqkv_cls = linear_cls if not self.return_residual else linear_resid_cls
        inner_attn_cls = None
        if use_aiter_attn:
            inner_attn_cls = AITERSelfAttention
        elif use_hip_attn:
            inner_attn_cls = HipSelfAttention
        else:
            inner_attn_cls = SelfAttention


        inner_cross_attn_cls = None
        if use_aiter_attn:
            inner_cross_attn_cls = AITERCrossAttention
        elif use_hip_attn:
            inner_cross_attn_cls = HipCrossAttention
        else:
            inner_cross_attn_cls = CrossAttention

        self.Wqkv = wqkv_cls(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.inner_attn = inner_attn_cls(
            causal=causal,
            softmax_scale=softmax_scale,
            attention_dropout=dropout,
        )
        self.inner_cross_attn = inner_cross_attn_cls(
            causal=causal, 
            softmax_scale=softmax_scale, 
            attention_dropout=dropout
        )
        
        self.out_proj = linear_cls(embed_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        return torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)"""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then do attention"""
        # TODO: this only uses seqlen_offset and not lengths_per_sample.
        kv = self._update_kv_cache(kv, inference_params)
        return self.inner_cross_attn(q, kv)

    def forward(
        self,
        x,
        x_kv=None,
        key_padding_mask=None,
        cu_seqlens=None,
        max_seqlen=None,
        mixer_subset=None,
        inference_params=None,
        **kwargs,
    ):
        """
        Arguments:
            x: (batch, seqlen, hidden_dim) (where hidden_dim = num heads * head dim) if
                cu_seqlens is None and max_seqlen is None, else (total, hidden_dim) where total
                is the is the sum of the sequence lengths in the batch.
            x_kv: (batch, seqlen, hidden_dim), only applicable for cross-attention. If None, use x.
            cu_seqlens: (batch_size + 1,), dtype torch.int32. The cumulative sequence lengths
                of the sequences in the batch, used to index into x. Only applicable when using
                FlashAttention.
            max_seqlen: int. Maximum sequence length in the batch.
            key_padding_mask: boolean mask, True means to keep, False means to mask out.
                (batch, seqlen). Only applicable when not using FlashAttention.
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
            inference_params: for generation. Adapted from Megatron-LM (and Apex)
            https://github.com/NVIDIA/apex/blob/3ff1a10f72ec07067c4e44759442329804ac5162/apex/transformer/testing/standalone_transformer_lm.py#L470
        """
        if cu_seqlens is not None:
            assert max_seqlen is not None
            assert key_padding_mask is None
            assert self.rotary_emb_dim == 0
        if key_padding_mask is not None:
            assert cu_seqlens is None
            assert max_seqlen is None
        if inference_params is not None:
            assert key_padding_mask is None
            assert cu_seqlens is None and max_seqlen is None

        kwargs = ( {"key_padding_mask": key_padding_mask, **kwargs} )
        seqlen_offset = (
            0
            if inference_params is None
            else (
                inference_params.lengths_per_sample
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
        )
        rotary_max_seqlen = inference_params.max_seqlen if inference_params is not None else None
        if self.num_heads_kv == self.num_heads:
            assert x_kv is None and mixer_subset is None
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            qkv = rearrange(qkv, "... (three h d) -> ... three h d", three=3, d=self.head_dim)
            if self.rotary_emb_dim > 0:
                qkv = self.rotary_emb(
                    qkv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )
            if inference_params is None:
                if not self.checkpointing:
                    context = self.inner_attn(qkv, **kwargs)
                else:
                    context = torch.utils.checkpoint.checkpoint(self.inner_attn, qkv, **kwargs)
            else:
                context = self._update_kvcache_attention(
                    qkv[:, :, 0], qkv[:, :, 1:], inference_params
                )
        else:
            
            assert self.num_heads_kv != self.num_heads
            if not self.return_residual:
                qkv = self.Wqkv(x)
            else:
                qkv, x = self.Wqkv(x)
            q = qkv[..., : self.num_heads * self.head_dim]
            kv = qkv[..., self.num_heads * self.head_dim :]
            q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
            kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)

            if self.rotary_emb_dim > 0:
                q, kv = self.rotary_emb(
                    q, kv, seqlen_offset=seqlen_offset, max_seqlen=rotary_max_seqlen
                )
            if inference_params is None:
                if not self.checkpointing:
                    context = self.inner_cross_attn(q, kv, **kwargs)
                else:
                    context = torch.utils.checkpoint.checkpoint(
                        self.inner_cross_attn, q, kv, **kwargs
                    )
            else:
                context = self._update_kvcache_attention(q, kv, inference_params)
        out = self.out_proj(rearrange(context, "... h d -> ... (h d)"))
        return out if not self.return_residual else (out, x)

