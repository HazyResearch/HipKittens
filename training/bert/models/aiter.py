import torch
import torch.nn as nn
from typing import Optional
import aiter

class AITERBertSelfAttention(nn.Module):
    """
    Uses aiter.flash_attn_func when there is NO padding.
    Falls back to MHA-style expansion if num_key_value_heads < num_attention_heads (GQA).
    Expects HF additive mask: [B,1,1,N] (0 keep, -inf mask)
    """
    def __init__(self, config, layer_idx=None, deterministic: bool = False):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError("hidden_size must be multiple of num_attention_heads")
        
        self.layer_idx = layer_idx
        self.num_attention_heads = config.num_attention_heads                  # h_q
        self.num_key_value_heads = getattr(config, "num_key_value_heads", self.num_attention_heads)  # h_kv
        self.attention_head_size = config.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # Linear layers for Q, K, V
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        self.value = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size)
        
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.deterministic = deterministic
        self.is_causal = False

        print(f"AITER BertSelfAttention layer {layer_idx}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[tuple] = None,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H = self.num_attention_heads
        HKV = self.num_key_value_heads
        D = self.attention_head_size

        q = self.query(hidden_states).view(B, N, H, D).to(torch.bfloat16).contiguous()
        k = self.key(hidden_states).view(B, N, HKV, D).to(torch.bfloat16).contiguous()
        v = self.value(hidden_states).view(B, N, HKV, D).to(torch.bfloat16).contiguous()

        p = float(self.dropout.p) if self.training else 0.0
        out_bnhd, softmax_lse = aiter.flash_attn_func(
            q, k, v,
            dropout_p=p,
            causal=self.is_causal,
            return_lse=True,
            deterministic=self.deterministic,
        )

        ctx = out_bnhd.to(q.dtype).contiguous().view(B, N, H * D)
        return ctx, None
